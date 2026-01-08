# overlay_demo.py
# 원본 영상 위에 bbox(+track id) + LSTM 이진분류 결과를 오버레이해서 mp4로 저장
#
# 수정사항:
# - 객체 이동 궤적(Trajectory) 표시 제거
# - MediaPipe 스켈레톤 Live 추론 기능 추가 (전처리 없는 정확한 오버레이)

import argparse
import json
import os
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

# MediaPipe Pose Connections
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
]

def _id_to_key(stable_id:int)->str:
    return f"id_{stable_id:04d}"


def _load_results(path:str)->dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data=json.load(f)
    out={}
    if isinstance(data, dict):
        for k, v in data.items():
            out[str(k)]=v
        return out
    for r in data:
        out[str(r.get("id"))]=r
    return out


def _analysis_windows_from_csv(df:pd.DataFrame, start_offset:int=140, end_offset:int=40)->dict:
    win={}
    g=df.groupby("stable_id")["frame"].max()
    for sid, maxf in g.items():
        maxf=int(maxf)
        win[int(sid)]=(maxf-start_offset, maxf-end_offset, maxf)
    return win


def _draw_corner_box(img, x1, y1, x2, y2, color, thickness=2, corner=18):
    x1=int(x1); y1=int(y1); x2=int(x2); y2=int(y2)
    cv2.line(img, (x1, y1), (x1+corner, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1+corner), color, thickness)
    cv2.line(img, (x2, y1), (x2-corner, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1+corner), color, thickness)
    cv2.line(img, (x1, y2), (x1+corner, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2-corner), color, thickness)
    cv2.line(img, (x2, y2), (x2-corner, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2-corner), color, thickness)


def _alpha_rect(img, x1, y1, x2, y2, color, alpha=0.55):
    x1=int(x1); y1=int(y1); x2=int(x2); y2=int(y2)
    x1=max(0, x1); y1=max(0, y1)
    x2=min(img.shape[1]-1, x2); y2=min(img.shape[0]-1, y2)
    if x2<=x1 or y2<=y1:
        return
    sub = img[y1:y2, x1:x2]
    rect = np.full(sub.shape, color, dtype=np.uint8)
    res = cv2.addWeighted(sub, 1-alpha, rect, alpha, 0)
    img[y1:y2, x1:x2] = res


def _put_label(img, x, y, title, subtitle=None, color_bg=(0,0,0), color_fg=(255,255,255), progress=None):
    font=cv2.FONT_HERSHEY_SIMPLEX
    scale=0.55
    thickness=1

    (tw, th), _=cv2.getTextSize(title, font, scale, thickness+1)
    subw=subh=0
    if subtitle:
        (subw, subh), _=cv2.getTextSize(subtitle, font, 0.45, 1)

    pad=6
    w=max(tw, subw)+pad*2
    h=th+pad*2+(subh+4 if subtitle else 0)+(8 if progress is not None else 0)

    x1=x
    y1=y-h
    x2=x+w
    y2=y

    _alpha_rect(img, x1, y1, x2, y2, color_bg, alpha=0.7)
    cv2.putText(img, title, (x1+pad, y1+pad+th), font, scale, color_fg, thickness+1, cv2.LINE_AA)
    if subtitle:
        cv2.putText(img, subtitle, (x1+pad, y1+pad+th+subh+4), font, 0.45, color_fg, 1, cv2.LINE_AA)

    if progress is not None:
        px1=x1+pad
        px2=x2-pad
        py2=y2-pad
        py1=py2-4
        cv2.rectangle(img, (px1, py1), (px2, py2), (200,200,200), 1)
        fill=int(px1+(px2-px1)*float(np.clip(progress, 0.0, 1.0)))
        cv2.rectangle(img, (px1, py1), (fill, py2), (255,255,255), -1)


def _draw_skeleton(img, landmarks, x1, y1, w, h, canvas_size, color=(255, 255, 255)):
    """
    NPZ의 정규화된 좌표를 원본 이미지의 bbox 영역으로 매핑하여 스켈레톤을 그립니다.
    """
    cw, ch = canvas_size
    # resize_with_padding 역산
    scale = min(cw / w, ch / h)
    offset_x = (cw - w * scale) / 2
    offset_y = (ch - h * scale) / 2

    points = []
    for lm in landmarks:
        # lm[0], lm[1]은 canvas(cw, ch) 상의 0~1 좌표
        lx_canvas = lm[0] * cw
        ly_canvas = lm[1] * ch
        
        # canvas -> original crop resolution
        lx_crop = (lx_canvas - offset_x) / scale
        ly_crop = (ly_canvas - offset_y) / scale
        
        # crop -> full frame
        fx = int(x1 + lx_crop)
        fy = int(y1 + ly_crop)
        points.append((fx, fy))

    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx < len(points) and end_idx < len(points):
            p1, p2 = points[start_idx], points[end_idx]
            # 좌표가 영상 범위 내에 있는지 대략 확인
            cv2.line(img, p1, p2, color, 1, cv2.LINE_AA)
            cv2.circle(img, p1, 2, color, -1, cv2.LINE_AA)
            cv2.circle(img, p2, 2, color, -1, cv2.LINE_AA)


def _draw_hud(img, frame_idx, total_frames, stats, parking_status="Unouccupied"):
    h, w = img.shape[:2]
    _alpha_rect(img, 0, 0, w, 80, (20, 20, 20), alpha=0.8)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    # System Status
    cv2.putText(img, "GAIT Abnormality Detection SYSTEM", (20, 35), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Stats
    sx = w - 350
    cv2.putText(img, f"TOTAL TRACKS: {stats['total']}", (sx, 25), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"NORMAL: {stats['normal']}", (sx + 150, 25), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(img, f"ABNORMAL: {stats['abnormal']}", (sx + 150, 48), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    
    # Parking Status
    cv2.putText(img, f"PARKING: {parking_status}", (20, 65), font, 0.6, (255, 255, 0), 1, cv2.LINE_AA)
    
    # Progress bar at the very top
    prog_w = int(w * (frame_idx / total_frames))
    cv2.rectangle(img, (0, 0), (prog_w, 3), (0, 165, 255), -1)


def _ema(prev, cur, a=0.75):
    if prev is None:
        return cur
    return (a*np.array(prev)+(1-a)*np.array(cur)).tolist()


def render(video_path:str, bboxes_csv:str, results_json:str, out_path:str, only_person=True, smooth=True, live_skeleton=True):
    df=pd.read_csv(bboxes_csv)
    df.rename(columns={"class": "cls_name"}, inplace=True)

    # [FIX] 분석 구간은 추론 대상인 'person' 클래스에 대해서만 계산해야 합니다.
    # 기존 코드는 person과 car를 모두 포함하여 max_frame을 계산해 window가 잘못 설정되었습니다.
    person_df = df[df["cls_name"]=="person"].copy()
    windows=_analysis_windows_from_csv(person_df)

    if only_person:
        df = person_df # only_person=True이면, 이후 로직에서 사용할 df를 person_df로 교체
    
    df["x2"]=df["x"]+df["w"]
    df["y2"]=df["y"]+df["h"]

    by_frame=defaultdict(list)
    for r in df.itertuples(index=False):
        by_frame[int(r.frame)].append({
            "stable_id":int(r.stable_id),
            "cls_name": r.cls_name,
            "x1":int(r.x),
            "y1":int(r.y),
            "x2":int(r.x2),
            "y2":int(r.y2),
            "w":int(r.w),
            "h":int(r.h),
            "conf":float(r.conf),
            "is_parked": int(r.is_parked) if hasattr(r, "is_parked") else 0
        })

    results=_load_results(results_json)
    # NPZ 데이터 로드 (live_skeleton=False일 때만)
    skeletons = {}
    if not live_skeleton:
        inference_dir = os.path.dirname(results_json)
        for sid in df["stable_id"].unique():
            npz_p = os.path.join(inference_dir, f"id_{sid:04d}.npz")
            if os.path.exists(npz_p):
                try:
                    data = np.load(npz_p)
                    skeletons[sid] = {
                        "landmarks": data["image_landmarks"],
                        "valid": data["valid"],
                        "canvas_size": (int(data["width"][0]), int(data["height"][0]))
                    }
                except: pass
    
    # Live MediaPipe 초기화
    mp_pose = None
    if live_skeleton:
        mp_pose = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )

    cap=cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps=cap.get(cv2.CAP_PROP_FPS) or 30.0
    W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nframes=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc=cv2.VideoWriter_fourcc(*"mp4v")
    out=cv2.VideoWriter(out_path, fourcc, fps, (W,H))

    smooth_bbox={}
    last_seen={}
    # trajectories 제거됨
    
    stats = {"total": 0, "normal": 0, "abnormal": 0}
    seen_ids = set()

    vid_idx=0
    while True:
        ok, frame=cap.read()
        if not ok:
            break
        vid_idx+=1
        frame_csv=vid_idx

        dets=by_frame.get(frame_csv, [])
        
        # 현재 프레임 주차 상태 확인
        current_parking_status = "Empty"
        for d in dets:
            if d.get("is_parked", 0) == 1:
                current_parking_status = "Parked"
                break

        for d in dets:
            sid=d["stable_id"]
            cls_name=d["cls_name"]
            
            # 차량 등 비사람 객체 처리 (화면 표시용)
            if cls_name != "person":
                # 주차된 차만 표시하거나, 필요에 따라 로직 추가 가능
                # 여기서는 HUD에 상태를 띄우므로 박스는 그리지 않거나, 주차된 차만 그릴 수 있음
                # (이전 요청사항에 따라 주차된 차만 그리는 로직 유지/추가 가능)
                if d.get("is_parked", 0) == 1:
                     _draw_corner_box(frame, d["x1"], d["y1"], d["x2"], d["y2"], (255, 0, 0), thickness=2, corner=20)
                     _put_label(frame, d["x1"], max(25, d["y1"]-8), "PARKED", color_bg=(100, 0, 0))
                continue

            key=_id_to_key(sid)
            
            if sid not in seen_ids:
                seen_ids.add(sid)
                stats["total"] += 1

            x1,y1,x2,y2 = d["x1"], d["y1"], d["x2"], d["y2"]
            orig_w, orig_h = d["w"], d["h"]
            
            if smooth:
                prev=smooth_bbox.get(sid)
                sm=_ema(prev, [x1,y1,x2,y2], a=0.78)
                smooth_bbox[sid]=sm
                x1,y1,x2,y2=map(int, sm)

            # Trajectory 업데이트 제거됨

            last_seen[sid]=frame_csv

            # 결과 조회 및 라벨 결정 로직 수정
            rr = results.get(key)
            
            # 분석 윈도우 정보 가져오기 (s: 시작, e: 끝)
            s, e = 0, 0
            has_window = False
            if sid in windows:
                s, e, _ = windows[sid]
                has_window = True

            # 상태 결정
            state = "WAITING"
            if has_window:
                if frame_csv < s:
                    state = "WAITING"
                elif frame_csv <= e:
                    state = "ANALYZING"
                else:
                    state = "RESULT"
            
            # 상태별 변수 설정
            label = "WAITING..."
            subtitle = "0.0s"
            color = (200, 200, 200) # 회색
            bg = (50, 50, 50)
            prog = None
            
            is_ab = False
            prob_val = 0.0

            if state == "ANALYZING":
                label = "ANALYZING..."
                color = (0, 215, 255) # 황금색
                bg = (20, 20, 20)
                # 시간 표시
                elapsed = max(0.0, (frame_csv - s) / fps)
                subtitle = f"{elapsed:.1f}s"
                # 진행률
                if e > s:
                    prog = (frame_csv - s) / float(e - s)

            elif state == "RESULT":
                if rr is not None:
                    is_ab = (int(rr.get("pred_class", 0)) == 1)
                    p_norm = float(rr.get("p_normal", 0.0))
                    p_ab = float(rr.get("p_abnormal", 0.0))
                    
                    if is_ab:
                        label = "ABNORMAL"
                        prob_val = p_ab
                        color = (0, 0, 255) # 빨간색
                        bg = (0, 0, 100)
                        _draw_corner_box(frame, x1-2, y1-2, x2+2, y2+2, (0,0,255), 1)
                    else:
                        label = "NORMAL"
                        prob_val = p_norm
                        color = (0, 255, 0) # 초록색
                        bg = (0, 80, 0)
                    
                    subtitle = f"{prob_val:.1%}"
                    
                    # 통계 업데이트
                    if f"stat_{sid}" not in seen_ids:
                        seen_ids.add(f"stat_{sid}")
                        if is_ab: stats["abnormal"] += 1
                        else: stats["normal"] += 1
                else:
                    label = "UNKNOWN"
                    subtitle = ""

            # BBox
            _draw_corner_box(frame, x1, y1, x2, y2, color, thickness=2, corner=20)
            
            # Skeleton (Only during analysis window)
            if state == "ANALYZING":
                if live_skeleton and mp_pose:
                    # 현재 bbox crop에서 바로 추론
                    h_img, w_img = frame.shape[:2]
                    cx1, cy1 = max(0, x1), max(0, y1)
                    cx2, cy2 = min(w_img, x2), min(h_img, y2)
                    
                    if cx2 > cx1 and cy2 > cy1:
                        crop = frame[cy1:cy2, cx1:cx2]
                        # Convert to RGB
                        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        res = mp_pose.process(crop_rgb)
                        if res.pose_landmarks:
                            # Normalize landmarks -> list of (x,y)
                            lms = []
                            for lm in res.pose_landmarks.landmark:
                                lms.append((lm.x, lm.y))
                            
                            cw_crop = cx2 - cx1
                            ch_crop = cy2 - cy1
                            _draw_skeleton(frame, lms, cx1, cy1, cw_crop, ch_crop, (cw_crop, ch_crop), color)

                else:
                    # NPZ 기반 (기존 로직)
                    if sid in skeletons:
                        s_data = skeletons[sid]
                        start_f = windows[sid][0]
                        local_idx = frame_csv - start_f
                        if 0 <= local_idx < len(s_data["landmarks"]):
                            if s_data["valid"][local_idx]:
                                _draw_skeleton(frame, s_data["landmarks"][local_idx], x1, y1, orig_w, orig_h, s_data["canvas_size"], color)

            title=f"{key} | {label}"
            
            py=max(25, y1-8)
            _put_label(frame, x1, py, title, subtitle=subtitle, color_bg=bg, progress=prog)

            # [NEW] 결과 화면에 크게 표시
            if state == "RESULT" and rr is not None:
                # 박스 중앙 계산
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                # 텍스트 구성
                l_text = f"{label}"
                l_sub = f"{prob_val:.1%}"
                
                font_scale = 1.0
                thickness = 2
                font = cv2.FONT_HERSHEY_SIMPLEX
                
                (tw, th), _ = cv2.getTextSize(l_text, font, font_scale, thickness)
                (sw, sh), _ = cv2.getTextSize(l_sub, font, 0.7, 1)
                
                # 배경 박스 크기
                bw = max(tw, sw) + 20
                bh = th + sh + 30
                
                bx1 = cx - bw // 2
                by1 = cy - bh // 2
                bx2 = bx1 + bw
                by2 = by1 + bh
                
                # 화면 밖으로 나가지 않게 클리핑은 생략(보통 ROI 내부이므로)
                _alpha_rect(frame, bx1, by1, bx2, by2, bg, alpha=0.8)
                
                # 텍스트 그리기
                cv2.putText(frame, l_text, (cx - tw // 2, by1 + 20 + th), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
                cv2.putText(frame, l_sub, (cx - sw // 2, by2 - 15), font, 0.7, (200, 200, 200), 1, cv2.LINE_AA)

        # HUD
        _draw_hud(frame, vid_idx, nframes, stats, parking_status=current_parking_status)

        if vid_idx%30==0:
            dead=[]
            for sid, lastf in last_seen.items():
                if frame_csv-lastf>45:
                    dead.append(sid)
            for sid in dead:
                last_seen.pop(sid, None)
                smooth_bbox.pop(sid, None)
                # trajectories.pop(sid, None) # Removed

        out.write(frame)

    cap.release()
    out.release()
    if mp_pose:
        mp_pose.close()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--bboxes", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--all", action="store_true", help="person만이 아니라 모든 class bbox를 그리고 싶으면 사용")
    ap.add_argument("--no_smooth", action="store_true")
    # Live Skeleton 옵션 추가 (기본값 True로 설정하여 사용자 요구 반영)
    ap.add_argument("--use_npz_skeleton", action="store_true", help="Live 추론 대신 기존 NPZ 파일 스켈레톤 사용")
    args=ap.parse_args()

    render(
        video_path=args.video,
        bboxes_csv=args.bboxes,
        results_json=args.results,
        out_path=args.out,
        only_person=(not args.all),
        smooth=(not args.no_smooth),

    )


if __name__=="__main__":
    main()
import os, csv
from collections import defaultdict
from typing import Optional, Set, List, Tuple
import cv2
import numpy as np
from ultralytics import YOLO
from utils import ensure_dir, clip_xyxy, color_for_person_vs_others
from id_fix import IDStabilizer

# Geometry utils
def bbox_center(b):
    x1,y1,x2,y2 = b
    return np.array([(x1+x2)/2, (y1+y2)/2])

def bbox_iou(a, b):
    ax1,ay1,ax2,ay2 = a
    bx1,by1,bx2,by2 = b
    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    iw,ih = max(0,ix2-ix1), max(0,iy2-iy1)
    inter = iw*ih
    area = (ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
    return inter/area if area>0 else 0

def get_padded_bbox(bbox, W, H, padding_ratio=0.2):
    """
    bbox에 padding_ratio만큼 여유 공간을 둡니다.
    padding_ratio=0.2 -> 상하좌우 20%씩 확장
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    
    pad_w = int(w * padding_ratio)
    pad_h = int(h * padding_ratio)
    
    nx1 = max(0, x1 - pad_w)
    ny1 = max(0, y1 - pad_h)
    nx2 = min(W, x2 + pad_w)
    ny2 = min(H, y2 + pad_h)
    
    return int(nx1), int(ny1), int(nx2), int(ny2)

# Main pipeline
def run_track_and_save(
    video_path: str,
    out_dir: str,
    roi_points: List[Tuple[int, int]] = None,
    model_path: str = "yolo11n.pt",
    tracker_yaml: str = "botsort.yaml",
    conf: float = 0.15,
    imgsz: int = 960,
    default_fps: float = 30.0,

    save_vis: bool = True,
    save_bbox_csv: bool = True,
    bbox_csv_name: str = "bboxes.csv",
    bbox_only_person: bool = True,

    save_crops: bool = True,
    save_only_classes: Optional[Set[str]] = None,
    crop_padding: float = 0.5,  # 크롭 여유 공간 비율

    enable_id_fix: bool = True,
    idfix_max_age: int = 30,
    idfix_use_appearance: bool = True,
):

    ensure_dir(out_dir)
    model = YOLO(model_path)

    stabilizer = IDStabilizer(idfix_max_age, use_appearance=idfix_use_appearance) if enable_id_fix else None

    # 상태 기록
    person_car_map = {}
    pair_scores = defaultdict(int)
    
    valid_person_ids = set() 
    valid_car_ids = set()    
    
    # 크롭 버퍼
    crop_buffer = defaultdict(list)
    saved_ids = set()

    roi_mask = None
    roi_poly = np.array(roi_points, np.int32) if roi_points else None

    vis_writer, frame_idx = None, 0

    # CSV 준비
    csv_f, csv_writer = None, None
    if save_bbox_csv:
        csv_path = os.path.join(out_dir, bbox_csv_name)
        csv_f = open(csv_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(csv_f, fieldnames=["frame","class","raw_id","stable_id","x","y","w","h","conf"])
        csv_writer.writeheader()

    try:
        for result in model.track(video_path, stream=True, conf=conf, imgsz=imgsz, tracker=tracker_yaml, verbose=False):

            clean_frame = result.orig_img  # YOLO 결과의 원본 이미지
            H, W = clean_frame.shape[:2]
            frame_idx += 1

            # 시각화용 프레임은 별도로 복사 (여기서만 그림 그리기)
            vis_frame = clean_frame.copy() if save_vis else None

            # ROI 마스크 생성
            if roi_points and roi_mask is None:
                roi_mask = np.zeros((H, W), dtype=np.uint8)
                cv2.fillPoly(roi_mask, [roi_poly], 1)

            # 시각화 초기화
            if vis_writer is None and save_vis:
                vis_path = os.path.join(out_dir, "visualized.mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                vis_writer = cv2.VideoWriter(vis_path, fourcc, default_fps, (W, H))

            if roi_poly is not None and vis_frame is not None:
                cv2.polylines(vis_frame, [roi_poly], True, (0, 255, 255), 2)

            if result.boxes is None or result.boxes.id is None:
                if vis_writer: vis_writer.write(vis_frame)
                continue

            ids = result.boxes.id.cpu().numpy().astype(int)
            xyxy = result.boxes.xyxy.cpu().numpy()
            clses = result.boxes.cls.cpu().numpy().astype(int)
            names = result.names
            confs = result.boxes.conf.cpu().numpy()

            raw_dets = []
            for i,(rid,cls,(x1,y1,x2,y2)) in enumerate(zip(ids,clses,xyxy)):
                name = names[int(cls)]
                box = clip_xyxy(x1,y1,x2,y2,W,H)
                if box:
                    raw_dets.append({"class":name,"raw_id":rid,"bbox":box,"conf":float(confs[i])})

            # 0. 사전 필터링 & ID Fix
            dets = []
            if roi_mask is not None:
                for d in raw_dets:
                    if d["class"] == "person":
                        dets.append(d)
                        continue
                    
                    if d["class"] in ["car", "truck", "bus"]:
                        cx1, cy1, cx2, cy2 = d["bbox"]
                        h_box = cy2 - cy1
                        foot_y1 = int(cy2 - h_box * 0.25)
                        foot_y2 = int(cy2)
                        foot_x1, foot_x2 = int(cx1), int(cx2)
                        foot_y1 = max(0, foot_y1); foot_y2 = min(H, foot_y2)
                        foot_x1 = max(0, foot_x1); foot_x2 = min(W, foot_x2)

                        foot_area = (foot_x2 - foot_x1) * (foot_y2 - foot_y1)
                        if foot_area > 0:
                            overlap_area = np.sum(roi_mask[foot_y1:foot_y2, foot_x1:foot_x2])
                            if (overlap_area / foot_area) > 0.3:
                                dets.append(d)
                                valid_car_ids.add(d["raw_id"])
            else:
                dets = raw_dets
                for d in dets:
                    if d["class"] in ["car","truck","bus"]:
                        valid_car_ids.add(d["raw_id"])

            # ID Fix
            if stabilizer: dets = stabilizer.assign(frame_idx, clean_frame, dets, W, H)

            persons = [d for d in dets if d["class"]=="person"]
            cars = [d for d in dets if d["class"] in ["car","truck","bus"]] 

            # 1. 사람-차량 매칭
            for p in persons:
                pid = p["stable_id"]
                if pid in valid_person_ids: continue

                pb = p["bbox"]
                pc = bbox_center(pb)
                best_score, best_car = 0, None

                for c in cars:
                    cid = c["raw_id"]
                    cb = c["bbox"]
                    cc = bbox_center(cb)

                    dist = np.linalg.norm(pc-cc)
                    s_dist = 1/(dist+1e-6)
                    s_iou = bbox_iou(pb,cb)

                    pair_scores[(pid,cid)] += 1
                    s_time = pair_scores[(pid,cid)] / 10
                    score = 0.4*s_iou + 0.3*s_dist + 0.3*s_time

                    if score > best_score:
                        best_score, best_car = score, cid

                if best_car and best_score > 0.6:
                    person_car_map[pid] = best_car
                    print(f"EVENT: Person {pid} exited ROI Car {best_car}")
                    valid_person_ids.add(pid) 

            # 2. 크롭 저장
            if save_crops:
                for d in dets:
                    cls, rid = d["class"], d["raw_id"]
                    sid = d.get("stable_id", rid)
                    
                    # 확장 bbox 계산
                    nx1, ny1, nx2, ny2 = get_padded_bbox(d["bbox"], W, H, crop_padding)
                    
                    crop_img = clean_frame[ny1:ny2, nx1:nx2]
                    
                    if crop_img.size == 0: continue

                    is_valid_now = False
                    if cls in ["car", "truck", "bus"]:
                        is_valid_now = True
                    elif cls == "person":
                        is_valid_now = True
                    
                    if is_valid_now:
                        folder = os.path.join(out_dir, cls, f"id_{sid:04d}")
                        
                        # 버퍼 털기
                        if sid in crop_buffer:
                            ensure_dir(folder)
                            for old_fidx, old_crop in crop_buffer[sid]:
                                cv2.imwrite(os.path.join(folder, f"{old_fidx:06d}.jpg"), old_crop)
                            del crop_buffer[sid]
                            saved_ids.add(sid)

                        # 현재 프레임 저장
                        ensure_dir(folder)
                        cv2.imwrite(os.path.join(folder, f"{frame_idx:06d}.jpg"), crop_img)
                        saved_ids.add(sid)
                        
                    else:
                        # 사람만만 버퍼에 저장 (사람만)
                        if cls == "person":
                            crop_buffer[sid].append((frame_idx, crop_img.copy())) 
                            if len(crop_buffer[sid]) > 1000: 
                                crop_buffer[sid].pop(0)

            # 3. CSV 및 시각화
            filtered_dets = []
            for d in dets:
                cls, sid = d["class"], d.get("stable_id", d["raw_id"])
                if (cls in ["car", "truck", "bus"]) or (cls == "person" and sid in valid_person_ids):
                    filtered_dets.append(d)

            for d in filtered_dets:
                cls, rid = d["class"], d["raw_id"]
                sid = d.get("stable_id", rid)
                x1,y1,x2,y2 = d["bbox"]

                if csv_writer and ((not bbox_only_person) or cls=="person"):
                    csv_writer.writerow({"frame":frame_idx,"class":cls,"raw_id":rid,"stable_id":sid,
                                         "x":x1,"y":y1,"w":x2-x1,"h":y2-y1,"conf":d["conf"]})

                if vis_writer and vis_frame is not None:
                    color = color_for_person_vs_others(cls)
                    cv2.rectangle(vis_frame,(x1,y1),(x2,y2),color,2)
                    cv2.putText(vis_frame,f"{cls} {sid}",(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

            if vis_writer and vis_frame is not None: 
                vis_writer.write(vis_frame)

    finally:
        if vis_writer: vis_writer.release()
        if csv_f: csv_f.close()

    print("Processing complete.")
import os
import sys
import cv2
import numpy as np
import urllib.request
import mediapipe as mp

# 모델 경로는 고정
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
MODEL_PATH = "pose_landmarker_heavy.task" 

def ensure_model(model_url: str, model_path: str) -> None:
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
        return
    print(f"Downloading model -> {model_path}")
    urllib.request.urlretrieve(model_url, model_path)
    print("Download complete.")

def process_video(video_path, out_npz_path):
    """
    외부에서 호출 가능한 함수로 변경
    """
    if not os.path.exists(video_path):
        print(f"[Error] Video not found: {video_path}")
        return False

    ensure_model(MODEL_URL, MODEL_PATH)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Error] Failed to open: {video_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )

    img_list, world_list, valid_list, time_ms_list = [], [], [], []

    with PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok: break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            timestamp_ms = int(round(frame_idx * 1000.0 / fps))
            time_ms_list.append(timestamp_ms)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if not result.pose_landmarks:
                img_list.append(np.full((33, 5), np.nan, dtype=np.float32))
                world_list.append(np.full((33, 5), np.nan, dtype=np.float32))
                valid_list.append(False)
            else:
                pose_lms = result.pose_landmarks[0]
                pose_w_lms = result.pose_world_landmarks[0]

                img_arr = np.zeros((33, 5), dtype=np.float32)
                w_arr = np.zeros((33, 5), dtype=np.float32)

                for i, lm in enumerate(pose_lms):
                    img_arr[i] = [lm.x, lm.y, lm.z, getattr(lm,"visibility",np.nan), getattr(lm,"presence",np.nan)]
                for i, lm in enumerate(pose_w_lms):
                    w_arr[i] = [lm.x, lm.y, lm.z, getattr(lm,"visibility",np.nan), getattr(lm,"presence",np.nan)]

                img_list.append(img_arr)
                world_list.append(w_arr)
                valid_list.append(True)

            frame_idx += 1

    cap.release()

    if not img_list:
        print(f"[Warning] No frames processed for {video_path}")
        return False

    # 저장
    os.makedirs(os.path.dirname(out_npz_path), exist_ok=True)
    np.savez_compressed(
        out_npz_path,
        image_landmarks=np.stack(img_list, axis=0),
        world_landmarks=np.stack(world_list, axis=0),
        valid=np.array(valid_list, dtype=np.bool_),
        time_ms=np.array(time_ms_list, dtype=np.int64),
        fps=np.array([fps], dtype=np.float32),
        width=np.array([width], dtype=np.int32),
        height=np.array([height], dtype=np.int32),
        video_path=np.array([video_path]),
    )
    print(f"[MediaPipe] Saved: {out_npz_path}")
    return True
import os
import sys
import numpy as np
import pandas as pd


# 모델 입력 고정값
FPS = 30
TARGET_T = 90

# 전처리 하이퍼파라미터
EXTREME_ABS_THRESH = 10.0
BONE_MIN, BONE_MAX = 0.05, 1.20
USE_SMOOTHING = False
SMOOTH_WIN = 5


def moving_average_1d(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    pad = win // 2
    xpad = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(win, dtype=np.float32) / win
    return np.convolve(xpad, kernel, mode="valid")


def smooth_sequence(sel: np.ndarray, win: int) -> np.ndarray:
    # sel: (T,V,3)
    T, V, C = sel.shape
    out = sel.copy()
    for v in range(V):
        for c in range(C):
            out[:, v, c] = moving_average_1d(out[:, v, c], win)
    return out


def compute_valid_mask(sel: np.ndarray) -> np.ndarray:
    """
    sel: (T,11,3)
    valid 조건:
      - finite
      - 폭발값 없음
      - 다리뼈 4개 중 2개 이상이 [BONE_MIN, BONE_MAX] 범위
    """
    finite = np.isfinite(sel).all(axis=(1, 2))
    extreme = (np.abs(sel) > EXTREME_ABS_THRESH).any(axis=(1, 2))

    # V=11 인덱스
    # 0:SpineBase, 1:ShoulderL, 2:ShoulderR, 3:HipL,4:KneeL,5:AnkleL,6:FootL, 7:HipR,8:KneeR,9:AnkleR,10:FootR
    bones = [(3, 4), (4, 5), (7, 8), (8, 9)]  # (Hip-Knee),(Knee-Ankle) 좌/우

    ok_count = np.zeros(sel.shape[0], dtype=np.int32)
    for a, b in bones:
        d = np.linalg.norm(sel[:, a, :] - sel[:, b, :], axis=1)
        ok = np.isfinite(d) & (d >= BONE_MIN) & (d <= BONE_MAX)
        ok_count += ok.astype(np.int32)

    bone_ok = ok_count >= 2
    return finite & (~extreme) & bone_ok


def interpolate_invalid(sel: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """
    sel: (T,V,3)
    valid_mask True 프레임을 기준으로 각 좌표를 선형보간
    """
    T = sel.shape[0]
    flat = sel.reshape(T, -1)
    idx = np.arange(T)

    if valid_mask.sum() == 0:
        return np.zeros_like(sel, dtype=np.float32)

    for k in range(flat.shape[1]):
        y = flat[:, k]
        good = valid_mask & np.isfinite(y)
        if good.sum() >= 2:
            flat[:, k] = np.interp(idx, idx[good], y[good])
        elif good.sum() == 1:
            flat[:, k] = y[good][0]
        else:
            flat[:, k] = 0.0

    return flat.reshape(T, sel.shape[1], 3).astype(np.float32)


def compute_scale_from_bones(sel: np.ndarray) -> float:
    bones = [(3, 4), (4, 5), (7, 8), (8, 9)]
    meds = []
    for a, b in bones:
        d = np.linalg.norm(sel[:, a, :] - sel[:, b, :], axis=1)
        ok = np.isfinite(d) & (d >= BONE_MIN) & (d <= BONE_MAX)
        if ok.sum() > 0:
            meds.append(float(np.median(d[ok])))

    if len(meds) == 0:
        return 1.0
    scale = float(np.mean(meds))
    if (not np.isfinite(scale)) or scale < 1e-6:
        scale = 1.0
    return scale


def compute_scale_from_hip(sel: np.ndarray) -> float:
    d = np.linalg.norm(sel[:, 3, :] - sel[:, 7, :], axis=1)
    ok = np.isfinite(d) & (d >= BONE_MIN) & (d <= BONE_MAX)
    if ok.sum() == 0:
        return 1.0
    scale = float(np.median(d[ok]))
    if (not np.isfinite(scale)) or scale < 1e-6:
        scale = 1.0
    return scale


def to_output_df(sel: np.ndarray) -> pd.DataFrame:
    T, V, _ = sel.shape
    cols = []
    for v in range(V):
        cols += [f"{v+1}_x", f"{v+1}_y", f"{v+1}_z"]
    df_out = pd.DataFrame(sel.reshape(T, V * 3), columns=cols)
    df_out.insert(0, "frame", np.arange(1, T + 1))
    return df_out


def transform_xyz_x_negz_negy(arr: np.ndarray) -> np.ndarray:
    """좌표계를 (x, y, z) -> (x, -z, -y)로 변환."""
    if arr.shape[-1] != 3:
        raise ValueError(f"Expected last dim=3 (x,y,z), got shape={arr.shape}")

    x = arr[..., 0]
    y = arr[..., 1]
    z = arr[..., 2]
    out = np.stack([-x, -z, -y], axis=-1)
    return out.astype(arr.dtype, copy=False)


# 외부 호출용 함수
def process_npz_to_csv(npz_path, csv_out_path):
    """
    NPZ 파일을 읽어 전처리 후 CSV로 저장합니다.
    성공 시 True, 실패 시 False 반환.
    """
    if not os.path.exists(npz_path):
        print(f"[Error] NPZ not found: {npz_path}")
        return False

    try:
        data = np.load(npz_path, allow_pickle=True)

        if "world_landmarks" not in data:
            print(f"[Skip] 'world_landmarks' key missing in {npz_path}")
            return False

        world = data["world_landmarks"]
        if world.ndim != 3 or world.shape[0] == 0:
            print(f"[Skip] Invalid data shape in {npz_path}")
            return False

        coords = world[:, :, :3].astype(np.float32)  # (T,33,3)

        # 1. 관절 선택 (MediaPipe Index)
        L_SHOULDER = 11
        R_SHOULDER = 12
        L_HIP = 23
        R_HIP = 24
        L_KNEE = 25
        R_KNEE = 26
        L_ANKLE = 27
        R_ANKLE = 28
        L_FOOT = 31
        R_FOOT = 32

        spine_base = (coords[:, L_HIP, :] + coords[:, R_HIP, :]) * 0.5  # (T,3)

        sel = np.stack(
            [
                spine_base,
                coords[:, L_SHOULDER, :],
                coords[:, R_SHOULDER, :],
                coords[:, L_HIP, :],
                coords[:, L_KNEE, :],
                coords[:, L_ANKLE, :],
                coords[:, L_FOOT, :],
                coords[:, R_HIP, :],
                coords[:, R_KNEE, :],
                coords[:, R_ANKLE, :],
                coords[:, R_FOOT, :],
            ],
            axis=1,
        )  # (T,11,3)

        # 2. 전처리 (보간 -> 센터링 -> 스케일링)
        valid = compute_valid_mask(sel)
        sel = interpolate_invalid(sel, valid)

        root = sel[:, 0:1, :]
        sel = sel - root

        scale = compute_scale_from_bones(sel)
        if (not np.isfinite(scale)) or scale < 1e-6:
            scale = compute_scale_from_hip(sel)
        sel = sel / scale

        if USE_SMOOTHING and SMOOTH_WIN > 1:
            sel = smooth_sequence(sel, SMOOTH_WIN)

        # 3. 길이 맞추기 (TARGET_T=90)
        T = sel.shape[0]
        if T < TARGET_T:
            pad = np.zeros((TARGET_T - T, sel.shape[1], 3), dtype=np.float32)
            sel = np.concatenate([sel, pad], axis=0)
        elif T > TARGET_T:
            sel = sel[:TARGET_T]

        # 4. 좌표계 변환 (Kinect v2 스타일)
        sel = transform_xyz_x_negz_negy(sel)

        # 5. 저장
        df_out = to_output_df(sel)
        os.makedirs(os.path.dirname(csv_out_path) or ".", exist_ok=True)
        df_out.to_csv(csv_out_path, index=False)

        print(f"[CSV] Saved: {csv_out_path}")
        return True

    except Exception as e:
        print(f"[Error] Failed to process csv: {e}")
        return False

if __name__ == "__main__":
    TEST_NPZ = "C:/Users/VICTUS/KHUDA_CV/video_npz/pose2.npz"
    TEST_CSV = "C:/Users/VICTUS/KHUDA_CV/inference/1.csv"
    process_npz_to_csv(TEST_NPZ, TEST_CSV)
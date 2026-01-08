from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2
import lap


@dataclass
class TrackState:
    stable_id: int
    last_frame: int
    bbox_xyxy: Tuple[int, int, int, int]
    hist: Optional[np.ndarray]


def _xyxy_to_cxcy(b):
    x1, y1, x2, y2 = b
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter + 1e-6)


def _hist_feature(frame, bbox):
    x1, y1, x2, y2 = bbox
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [24, 24], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def _hist_sim(h1, h2):
    if h1 is None or h2 is None:
        return 0.0
    return (cv2.compareHist(h1.astype(np.float32), h2.astype(np.float32), cv2.HISTCMP_CORREL) + 1) * 0.5


class IDStabilizer:

    def __init__(self, max_age=30, min_iou=0.1, max_center_dist_norm=0.2,
                 w_iou=0.55, w_dist=0.30, w_app=0.15, use_appearance=True):

        self.max_age = max_age
        self.min_iou = min_iou
        self.max_center_dist_norm = max_center_dist_norm
        self.w_iou = w_iou
        self.w_dist = w_dist
        self.w_app = w_app
        self.use_appearance = use_appearance

        self._next_id = 1
        self.tracks: Dict[int, TrackState] = {}
        self.raw_to_stable: Dict[int, int] = {}

    def _new_id(self):
        i = self._next_id
        self._next_id += 1
        return i

    def _purge(self, frame_idx):
        dead = [sid for sid, t in self.tracks.items() if frame_idx - t.last_frame > self.max_age]
        for sid in dead:
            del self.tracks[sid]

    def assign(self, frame_idx, frame, dets, W, H):
        self._purge(frame_idx)

        persons = [d for d in dets if d["class"] == "person"]
        if not persons:
            return dets

        diag = np.hypot(W, H)

        for d in persons:
            rid = d["raw_id"]
            if rid in self.raw_to_stable and self.raw_to_stable[rid] in self.tracks:
                sid = self.raw_to_stable[rid]
                d["stable_id"] = sid
                hist = _hist_feature(frame, d["bbox"]) if self.use_appearance else None
                self.tracks[sid] = TrackState(sid, frame_idx, d["bbox"], hist)

        unassigned = [d for d in persons if "stable_id" not in d]

        if not unassigned:
            return dets

        candidates = list(self.tracks.values())
        if not candidates:
            for d in unassigned:
                sid = self._new_id()
                d["stable_id"] = sid
                hist = _hist_feature(frame, d["bbox"]) if self.use_appearance else None
                self.tracks[sid] = TrackState(sid, frame_idx, d["bbox"], hist)
                self.raw_to_stable[d["raw_id"]] = sid
            return dets

        M, N = len(unassigned), len(candidates)
        cost = np.ones((M, N), dtype=np.float32)

        det_hists = [_hist_feature(frame, d["bbox"]) for d in unassigned]

        for i, d in enumerate(unassigned):
            cx1, cy1 = _xyxy_to_cxcy(d["bbox"])
            for j, t in enumerate(candidates):
                cx2, cy2 = _xyxy_to_cxcy(t.bbox_xyxy)
                dist = np.hypot(cx1 - cx2, cy1 - cy2) / diag
                if dist > self.max_center_dist_norm:
                    continue

                iou = _iou(d["bbox"], t.bbox_xyxy)
                if iou < self.min_iou:
                    continue

                dist_sim = 1 - dist / self.max_center_dist_norm
                app_sim = _hist_sim(det_hists[i], t.hist) if self.use_appearance else 0

                score = self.w_iou * iou + self.w_dist * dist_sim + self.w_app * app_sim
                cost[i, j] = 1 - score

        _, x, _ = lap.lapjv(cost, extend_cost=True)

        used = set()
        for i, d in enumerate(unassigned):
            j = int(x[i])
            if j < 0 or j >= N:
                continue
            if cost[i, j] > 0.65:
                continue

            sid = candidates[j].stable_id
            if sid in used:
                continue

            used.add(sid)
            d["stable_id"] = sid
            hist = det_hists[i]
            self.tracks[sid] = TrackState(sid, frame_idx, d["bbox"], hist)
            self.raw_to_stable[d["raw_id"]] = sid

        for d in unassigned:
            if "stable_id" in d:
                continue
            sid = self._new_id()
            d["stable_id"] = sid
            hist = _hist_feature(frame, d["bbox"]) if self.use_appearance else None
            self.tracks[sid] = TrackState(sid, frame_idx, d["bbox"], hist)
            self.raw_to_stable[d["raw_id"]] = sid

        return dets

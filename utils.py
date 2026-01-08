# utils.py
import os
from typing import Optional, Tuple

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def clip_xyxy(x1, y1, x2, y2, w: int, h: int) -> Optional[Tuple[int, int, int, int]]:
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2

def color_for_person_vs_others(class_name: str):
    if class_name == "person":
        return (0, 0, 255)  # red
    return (255, 0, 0)      # blue

def group_for_class(class_name: str) -> str:
    if class_name == "person":
        return "person"
    if class_name in {"car", "truck", "bus", "motorcycle", "bicycle"}:
        return "vehicle"
    return "other"

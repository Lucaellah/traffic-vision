"""YOLO 模型清單與載入。權重第一次使用時會自動下載到 weights/。"""

from __future__ import annotations

import os
from pathlib import Path

MODELS = {
    "YOLO11n（最快）": "yolo11n.pt",
    "YOLO11s（平衡）": "yolo11s.pt",
    "YOLO11m（較準）": "yolo11m.pt",
    "YOLO11n-seg（實例分割）": "yolo11n-seg.pt",
    "YOLO11s-seg（實例分割）": "yolo11s-seg.pt",
    # DOTA 空拍資料集訓練的旋轉框模型：一般模型認不出俯視角度的小車
    "YOLO11s-obb（空拍 / 俯視）": "yolo11s-obb.pt",
    "YOLO11m-obb（空拍 / 俯視，較準）": "yolo11m-obb.pt",
}

WEIGHTS_DIR = Path(
    os.environ.get("TRAFFIC_VISION_WEIGHTS", Path(__file__).resolve().parent.parent / "weights")
)


def weights_path(filename: str) -> Path:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    return WEIGHTS_DIR / filename


def load_model(filename: str):
    """載入 YOLO 模型；檔案不存在時 ultralytics 會下載到指定路徑。"""
    from ultralytics import YOLO

    return YOLO(str(weights_path(filename)))

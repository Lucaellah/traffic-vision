"""YOLO 模型清單與載入。權重第一次使用時會自動下載到 weights/。"""

from __future__ import annotations

import json
import os
from pathlib import Path

MODELS = {
    # YOLO26：不需要 NMS 的端到端架構，CPU 推論較快
    "YOLO26n（最快）": "yolo26n.pt",
    "YOLO26s（平衡）": "yolo26s.pt",
    "YOLO26m（較準）": "yolo26m.pt",
    "YOLO26s-seg（實例分割）": "yolo26s-seg.pt",
    # DOTA 空拍資料集訓練的旋轉框模型：一般模型認不出俯視角度的小車
    "YOLO26s-obb（空拍 / 俯視）": "yolo26s-obb.pt",
}

WEIGHTS_DIR = Path(
    os.environ.get("TRAFFIC_VISION_WEIGHTS", Path(__file__).resolve().parent.parent / "weights")
)
CUSTOM_DIR = "custom"


def weights_path(filename: str) -> Path:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    return WEIGHTS_DIR / filename


def custom_models() -> dict[str, str]:
    """weights/custom/ 裡微調好的模型（training/finetune.py 的輸出）。"""
    out = {}
    for pt in sorted((WEIGHTS_DIR / CUSTOM_DIR).glob("*.pt")):
        meta_file = pt.with_suffix(".json")
        meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
        data = meta.get("data")
        label = f"微調：{pt.stem}" + (f"（{data}）" if data else "")
        out[label] = f"{CUSTOM_DIR}/{pt.name}"
    return out


def available_models() -> dict[str, str]:
    """微調模型排在最前面：有的話通常就是使用者想用的。"""
    return {**custom_models(), **MODELS}


def load_model(filename: str):
    """載入 YOLO 模型；檔案不存在時 ultralytics 會下載到指定路徑。"""
    from ultralytics import YOLO
    from ultralytics.utils.events import events

    # 關閉 Ultralytics 的匿名使用統計（預設開啟），只影響這個行程，不改全域設定
    events.enabled = False
    return YOLO(str(weights_path(filename)))

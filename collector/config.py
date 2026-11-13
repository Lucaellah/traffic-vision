"""收集的對象、排程與存放位置。鏡頭編號取自高公局開放資料的 CCTV.xml。"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("TV_DATA_DIR", PROJECT_DIR / "data"))
RAW_DIR = DATA_DIR / "raw"

TZ = timezone(timedelta(hours=8))  # 台灣時間

# 國道 1 號南向 84K+600，湖口交流道 - 湖口服務區（畫面同時拍到南北向）
CAMERA = "18460"
STREAM_URL = "https://cctvn.freeway.gov.tw/abs2mjpg/bmjpg?camera={camera}"

# 尖峰時段每 5 分鐘錄一次，其他時段每 30 分鐘一次；每次 60 秒
PEAK_HOURS = ((6, 10), (16, 20))
PEAK_EVERY_MIN = 5
OFFPEAK_EVERY_MIN = 30
CLIP_SECONDS = 60


def now() -> datetime:
    return datetime.now(TZ)


def should_record(t: datetime) -> bool:
    """t 所在的 5 分鐘時段是否要錄影。timer 可能晚幾秒觸發，所以先對齊到 5 分鐘。"""
    minute = t.minute // PEAK_EVERY_MIN * PEAK_EVERY_MIN
    if any(start <= t.hour < end for start, end in PEAK_HOURS):
        return True
    return minute % OFFPEAK_EVERY_MIN == 0

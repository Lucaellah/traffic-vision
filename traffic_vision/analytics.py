"""軌跡資料集的整理與統計。也能讀原專案匯出的舊格式（Track ID / X / Y / Category）。"""

from __future__ import annotations

from pathlib import Path
from typing import IO

import numpy as np
import pandas as pd

REQUIRED = ("track_id", "x", "y")

# 欄位別名（比對時忽略大小寫、空白、底線）
ALIASES = {
    "trackid": "track_id",
    "id": "track_id",
    "category": "class_id",
    "class": "class_id",
    "classid": "class_id",
    "cls": "class_id",
    "classname": "class_name",
    "label": "class_name",
    "frame": "frame",
    "frameid": "frame",
    "times": "time_s",
    "time": "time_s",
    "x": "x",
    "y": "y",
    "w": "w",
    "h": "h",
    "conf": "conf",
    "confidence": "conf",
}


def _key(col: str) -> str:
    return "".join(ch for ch in str(col).lower() if ch.isalnum())


def read_table(file: str | Path | IO, name: str | None = None) -> pd.DataFrame:
    name = (name or str(file)).lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    if name.endswith(".csv"):
        return pd.read_csv(file)
    raise ValueError("只支援 CSV 或 Excel（.xlsx / .xls）檔案")


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """統一欄位名稱與型別；缺少必要欄位時丟出 ValueError。"""
    renamed = {}
    for col in df.columns:
        target = ALIASES.get(_key(col))
        if target and target not in renamed.values():
            renamed[col] = target
    out = df.rename(columns=renamed)

    missing = [c for c in REQUIRED if c not in out.columns]
    if missing:
        raise ValueError(
            f"資料缺少必要欄位：{', '.join(missing)}（需要 track_id、x、y，"
            f"目前欄位：{', '.join(map(str, df.columns))}）"
        )

    for col in ("track_id", "x", "y", "class_id", "frame", "time_s", "conf", "w", "h"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=list(REQUIRED)).copy()
    out["track_id"] = out["track_id"].astype(int)

    if "frame" not in out.columns:
        # 舊格式沒有幀號：以每個物件出現的順序代替
        out["frame"] = out.groupby("track_id").cumcount()
    if "class_name" not in out.columns:
        out["class_name"] = (
            out["class_id"].astype("Int64").astype(str) if "class_id" in out.columns else "unknown"
        )
    return out.sort_values(["track_id", "frame"]).reset_index(drop=True)


def class_counts(df: pd.DataFrame) -> pd.DataFrame:
    """各類別的不重複物件數。"""
    return (
        df.groupby("class_name")["track_id"].nunique()
        .rename("count").sort_values(ascending=False).reset_index()
    )


def track_summary(df: pd.DataFrame) -> pd.DataFrame:
    """每個物件的出現幀數、移動距離（像素）與平均速度。"""
    d = df.sort_values(["track_id", "frame"])
    step = np.hypot(d.groupby("track_id")["x"].diff(), d.groupby("track_id")["y"].diff())
    d = d.assign(step=step.fillna(0.0))

    agg = {
        "class_name": ("class_name", lambda s: s.mode().iat[0]),
        "points": ("frame", "size"),
        "first_frame": ("frame", "min"),
        "last_frame": ("frame", "max"),
        "distance_px": ("step", "sum"),
    }
    if "time_s" in d.columns:
        agg["duration_s"] = ("time_s", lambda s: s.max() - s.min())
    out = d.groupby("track_id").agg(**agg).reset_index()

    if "duration_s" in out.columns:
        dur = out["duration_s"].replace(0, np.nan)
        out["speed_px_s"] = (out["distance_px"] / dur).round(1)
        out["duration_s"] = out["duration_s"].round(2)
    out["distance_px"] = out["distance_px"].round(1)
    return out.sort_values("points", ascending=False).reset_index(drop=True)


def overview(df: pd.DataFrame) -> dict:
    info = {
        "rows": len(df),
        "tracks": int(df["track_id"].nunique()),
        "classes": int(df["class_name"].nunique()),
        "frames": int(df["frame"].nunique()),
    }
    if "time_s" in df.columns and df["time_s"].notna().any():
        info["duration_s"] = float(df["time_s"].max() - df["time_s"].min())
    return info

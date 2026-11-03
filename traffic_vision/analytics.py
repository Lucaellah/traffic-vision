"""軌跡資料集的整理與統計。也能讀原專案匯出的舊格式（Track ID / X / Y / Category）。"""

from __future__ import annotations

from pathlib import Path
from typing import IO

import numpy as np
import pandas as pd

from .labels import HEAVY_CATEGORIES, category_of

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


def track_classes(df: pd.DataFrame) -> pd.Series:
    """每個物件最常被判定的類別。

    同一台車在不同幀可能被判成 car 或 van，直接依類別數 track_id 會重複計算。
    """
    return df.groupby("track_id")["class_name"].agg(lambda s: s.mode().iat[0])


def class_counts(df: pd.DataFrame) -> pd.DataFrame:
    """各類別的物件數（每個物件只算一次）。"""
    return (
        track_classes(df).value_counts()
        .rename_axis("class_name").rename("count").reset_index()
    )


def track_summary(df: pd.DataFrame) -> pd.DataFrame:
    """每個物件的出現幀數、移動距離（像素）與平均速度。"""
    d = df.sort_values(["track_id", "frame"])
    step = np.hypot(d.groupby("track_id")["x"].diff(), d.groupby("track_id")["y"].diff())
    d = d.assign(step=step.fillna(0.0))

    agg = {
        "points": ("frame", "size"),
        "first_frame": ("frame", "min"),
        "last_frame": ("frame", "max"),
        "distance_px": ("step", "sum"),
    }
    if "time_s" in d.columns:
        agg["duration_s"] = ("time_s", lambda s: s.max() - s.min())
    out = d.groupby("track_id").agg(**agg)
    out.insert(0, "class_name", track_classes(d))
    out = out.reset_index()

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


# ---- 固定攝影機：越線、流量、車速 ----

def vehicle_crossings(crossings: pd.DataFrame, tracks: pd.DataFrame) -> pd.DataFrame:
    """計數線的越線紀錄，加上每台車的類別（統一類別名稱）。"""
    out = crossings[crossings["line"] == "count"].drop(columns="line")
    classes = track_classes(tracks)
    out = out.assign(class_name=out["track_id"].map(classes))
    out["category"] = out["class_name"].map(lambda n: category_of(n) if isinstance(n, str) else n)
    return out.reset_index(drop=True)


def hourly_rate(count: int, duration_s: float) -> float:
    """換算成每小時的數量。"""
    return count / duration_s * 3600 if duration_s > 0 else float("nan")


def heavy_share(categories: pd.Series) -> float:
    """大型車（公車、卡車）所占比例。"""
    return float(categories.isin(HEAVY_CATEGORIES).mean()) if len(categories) else float("nan")


def flow_series(crossings: pd.DataFrame, bin_s: float) -> pd.DataFrame:
    """每個時間區間各方向的越線車輛數；沒有車的區間補 0。"""
    if crossings.empty:
        return pd.DataFrame(columns=["start_s", "direction", "count"])
    bins = (crossings["time_s"] // bin_s * bin_s).rename("start_s")
    counts = crossings.groupby([bins, "direction"]).size().unstack(fill_value=0)
    full = np.arange(0, counts.index.max() + bin_s, bin_s)
    counts = counts.reindex(full, fill_value=0).rename_axis("start_s")
    return counts.reset_index().melt("start_s", var_name="direction", value_name="count")


def crossing_speeds(crossings: pd.DataFrame, distance_m: float,
                    max_kmh: float = 250) -> pd.DataFrame:
    """同一台車通過計數線與測速線的時間差 → 車速（km/h）。

    兩條線都要跨過才算得出來；時間差太短（換算超過 max_kmh）視為追蹤錯誤丟掉。
    """
    times = crossings.dropna(subset=["time_s"]).pivot_table(
        index="track_id", columns="line", values="time_s", aggfunc="first")
    if not {"count", "speed"} <= set(times.columns):
        return pd.DataFrame(columns=["track_id", "dt_s", "speed_kmh"])
    dt = (times["speed"] - times["count"]).abs().dropna()
    dt = dt[dt > 0]
    out = pd.DataFrame({"dt_s": dt, "speed_kmh": distance_m / dt * 3.6})
    return out[out["speed_kmh"] <= max_kmh].reset_index()


# ---- 行車紀錄器：擁擠度、弱勢用路人、碰撞時間 ----

def with_category(tracks: pd.DataFrame) -> pd.DataFrame:
    return tracks.assign(category=tracks["class_name"].map(category_of))


def presence_series(tracks: pd.DataFrame, bin_s: float = 1.0) -> pd.DataFrame:
    """每個時間區間「平均每幀畫面中有幾個」各類別物件。

    算的是同一時間畫面中的物件數，不受同一個物件被切成好幾個 ID 影響。
    """
    d = with_category(tracks)
    per_frame = d.groupby(["frame", "category"]).size().unstack(fill_value=0)
    frames = d.groupby("frame")["time_s"].first()
    per_frame = per_frame.reindex(frames.index, fill_value=0)
    bins = (frames // bin_s * bin_s).rename("start_s")
    out = per_frame.groupby(bins).mean().rename_axis(columns="category")
    return out.reset_index().melt("start_s", var_name="category", value_name="count")


def exposure(tracks: pd.DataFrame, total_frames: int, categories: list[str]) -> pd.Series:
    """各類別「畫面中有出現」的幀數占全部幀數的比例。"""
    d = with_category(tracks)
    frames = d[d["category"].isin(categories)].groupby("category")["frame"].nunique()
    return (frames / max(total_frames, 1)).reindex(categories, fill_value=0.0)


def ttc_events(tracks: pd.DataFrame, width: float, *, window_s: float = 0.5,
               threshold_s: float = 1.5, min_width_frac: float = 0.05,
               center_frac: float = 0.4, merge_s: float = 1.0) -> pd.DataFrame:
    """前方物件的碰撞時間（time to collision）低於門檻的事件。

    每個物件只留最危險的一次；間隔不到 merge_s 秒的事件視為同一次接近，只留 TTC 最小的。

    不需要知道真實距離：物件以固定相對速度接近時，框的大小 s 和距離成反比，
    TTC = s / (ds/dt) = 1 / (d ln s / dt)。用 window_s 內的點做線性回歸估斜率，降低框抖動的影響。
    只看畫面中央 center_frac 寬度內、框寬至少 min_width_frac 畫面寬的物件（前方、夠近）。
    """
    columns = ["track_id", "time_s", "frame", "class_name", "ttc_s", "x", "y", "w", "h"]
    rows = []
    d = tracks[(tracks["w"] > 0) & (tracks["h"] > 0)].sort_values(["track_id", "frame"])
    half = width * center_frac / 2
    for tid, g in d.groupby("track_id"):
        if len(g) < 3:
            continue
        t = g["time_s"].to_numpy()
        log_s = 0.5 * np.log(g["w"].to_numpy() * g["h"].to_numpy())
        best = None
        for i in range(len(g)):
            row = g.iloc[i]
            if abs(row["x"] - width / 2) > half or row["w"] < width * min_width_frac:
                continue
            sel = (t >= t[i] - window_s) & (t <= t[i])
            if sel.sum() < 3 or np.ptp(t[sel]) < window_s / 2:
                continue
            slope = np.polyfit(t[sel], log_s[sel], 1)[0]
            if slope <= 0:
                continue
            ttc = 1 / slope
            if ttc < threshold_s and (best is None or ttc < best[4]):
                best = (tid, t[i], int(row["frame"]), row["class_name"], ttc,
                        row["x"], row["y"], row["w"], row["h"])
        if best:
            rows.append(best)
    out = pd.DataFrame(rows, columns=columns).sort_values("time_s")
    group = (out["time_s"].diff() > merge_s).cumsum()
    out = out.loc[out.groupby(group)["ttc_s"].idxmin()] if len(out) else out
    out["category"] = out["class_name"].map(category_of)
    return out.reset_index(drop=True)

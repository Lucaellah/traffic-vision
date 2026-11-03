import io

import numpy as np
import pandas as pd
import pytest

from traffic_vision import analytics


@pytest.fixture
def tracks():
    return pd.DataFrame({
        "frame": [0, 1, 2, 0, 1],
        "time_s": [0.0, 0.5, 1.0, 0.0, 0.5],
        "track_id": [1, 1, 1, 2, 2],
        "class_name": ["car", "car", "car", "bus", "bus"],
        "x": [0.0, 3.0, 6.0, 10.0, 10.0],
        "y": [0.0, 4.0, 8.0, 0.0, 5.0],
    })


def test_normalize_legacy_format():
    legacy = pd.DataFrame({"Track ID": [3, 3], "X": [1, 2], "Y": [5, 6], "Category": [2, 2]})
    df = analytics.normalize(legacy)
    assert {"track_id", "x", "y", "class_id", "frame", "class_name"} <= set(df.columns)
    assert df["frame"].tolist() == [0, 1]
    assert df["class_name"].iat[0] == "2"


def test_normalize_missing_columns():
    with pytest.raises(ValueError, match="track_id"):
        analytics.normalize(pd.DataFrame({"x": [1], "y": [2]}))


def test_normalize_drops_bad_rows():
    df = analytics.normalize(pd.DataFrame({"track_id": [1, 2], "x": [1, "oops"], "y": [1, 2]}))
    assert df["track_id"].tolist() == [1]


def test_class_counts_unique_tracks(tracks):
    counts = analytics.class_counts(analytics.normalize(tracks)).set_index("class_name")["count"]
    assert counts.to_dict() == {"bus": 1, "car": 1}


def test_class_counts_track_counted_once_when_class_flips():
    # 同一台車被判成 car 兩次、van 一次：只算一台 car
    df = pd.DataFrame({"track_id": [7, 7, 7], "x": [0, 1, 2], "y": [0, 0, 0],
                       "class_name": ["car", "van", "car"]})
    counts = analytics.class_counts(analytics.normalize(df))
    assert counts.to_dict("records") == [{"class_name": "car", "count": 1}]


def test_track_summary_distance_and_speed(tracks):
    s = analytics.track_summary(analytics.normalize(tracks)).set_index("track_id")
    assert s.at[1, "distance_px"] == pytest.approx(10.0)  # 兩步各 5 px
    assert s.at[1, "speed_px_s"] == pytest.approx(10.0)
    assert s.at[2, "distance_px"] == pytest.approx(5.0)


def test_overview(tracks):
    ov = analytics.overview(analytics.normalize(tracks))
    assert ov == {"rows": 5, "tracks": 2, "classes": 2, "frames": 3, "duration_s": 1.0}


def test_read_table_csv(tracks):
    buf = io.BytesIO(tracks.to_csv(index=False).encode())
    assert len(analytics.read_table(buf, "a.csv")) == 5
    with pytest.raises(ValueError):
        analytics.read_table(buf, "a.json")


def crossings_df(rows):
    return pd.DataFrame(rows, columns=["line", "track_id", "direction", "time_s"])


def test_vehicle_crossings_uses_track_majority_class():
    tracks = pd.DataFrame({"track_id": [1, 1, 1, 2], "x": [0, 0, 0, 0], "y": [0, 0, 0, 0],
                           "class_name": ["car", "truck", "car", "bus"]})
    cr = crossings_df([("count", 1, "in", 1.0), ("count", 2, "out", 2.0),
                       ("speed", 1, "in", 1.5)])
    out = analytics.vehicle_crossings(cr, analytics.normalize(tracks))
    assert out[["track_id", "category"]].values.tolist() == [[1, "汽車"], [2, "公車"]]
    assert analytics.heavy_share(out["category"]) == 0.5


def test_hourly_rate():
    assert analytics.hourly_rate(12, 10) == 4320


def test_flow_series_fills_empty_bins():
    cr = crossings_df([("count", 1, "in", 1.0), ("count", 2, "out", 25.0)])
    flow = analytics.flow_series(cr, 10).set_index(["start_s", "direction"])["count"]
    assert flow[(10, "in")] == 0 and flow[(20, "out")] == 1 and flow[(0, "in")] == 1


def test_crossing_speeds():
    cr = crossings_df([("count", 1, "out", 2.0), ("speed", 1, "out", 3.0),   # 30 m / 1 s
                       ("count", 2, "out", 2.0),                              # 沒過測速線
                       ("count", 3, "in", 5.0), ("speed", 3, "in", 5.1)])     # 1080 km/h，丟掉
    speeds = analytics.crossing_speeds(cr, 30).set_index("track_id")["speed_kmh"]
    assert speeds.to_dict() == {1: pytest.approx(108.0)}


def test_presence_series_counts_objects_per_frame():
    df = pd.DataFrame({"frame": [0, 0, 1, 1, 1], "time_s": [0, 0, 0.5, 0.5, 0.5],
                       "track_id": [1, 2, 1, 2, 3], "x": [0] * 5, "y": [0] * 5,
                       "class_name": ["car", "person", "car", "person", "car"]})
    p = analytics.presence_series(analytics.normalize(df), 1.0).set_index("category")["count"]
    assert p["汽車"] == 1.5 and p["人"] == 1.0


def test_exposure():
    df = pd.DataFrame({"frame": [0, 1, 3], "track_id": [1, 1, 2], "x": [0] * 3, "y": [0] * 3,
                       "class_name": ["person", "person", "motorcycle"]})
    e = analytics.exposure(analytics.normalize(df), 4, ["人", "腳踏車", "機車"])
    assert e.to_dict() == {"人": 0.5, "腳踏車": 0.0, "機車": 0.25}


def approaching(track_id, t0, rate, x=500, frames=10, fps=10):
    """框大小以 exp(rate·t) 成長的物件：TTC = 1 / rate。"""
    t = np.arange(frames) / fps
    s = 60 * np.exp(rate * t)
    return pd.DataFrame({"track_id": track_id, "frame": (t0 + t) * fps, "time_s": t0 + t,
                         "class_name": "car", "x": x, "y": 500, "w": s, "h": s})


def test_ttc_events_detects_fast_approach_only():
    df = pd.concat([
        approaching(1, 0.0, rate=1.0),          # TTC 1 秒 → 事件
        approaching(2, 5.0, rate=0.2),          # TTC 5 秒 → 不算
        approaching(3, 9.0, rate=1.0, x=50),    # 在畫面邊緣 → 不算
    ])
    ev = analytics.ttc_events(analytics.normalize(df), width=1000)
    assert ev["track_id"].tolist() == [1]
    assert ev["ttc_s"].iat[0] == pytest.approx(1.0, rel=0.01)


def test_ttc_events_merge_same_moment():
    df = pd.concat([approaching(1, 0.0, rate=1.0), approaching(2, 0.2, rate=1.5)])
    ev = analytics.ttc_events(analytics.normalize(df), width=1000)
    assert ev["track_id"].tolist() == [2]

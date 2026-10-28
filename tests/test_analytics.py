import io

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

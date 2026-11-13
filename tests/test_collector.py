import io
from datetime import datetime

from collector import config
from collector.record import iter_jpegs


def at(hour: int, minute: int, second: int = 0) -> datetime:
    return datetime(2026, 11, 3, hour, minute, second, tzinfo=config.TZ)


def test_peak_records_every_slot():
    assert config.should_record(at(7, 35))
    assert config.should_record(at(19, 55, 4))  # timer 晚幾秒觸發也算同一個時段


def test_offpeak_only_on_half_hours():
    assert config.should_record(at(12, 0))
    assert config.should_record(at(12, 30, 3))
    assert not config.should_record(at(12, 5))
    assert config.should_record(at(10, 0))  # 尖峰剛結束，但整點仍是每 30 分鐘的時段
    assert not config.should_record(at(20, 15))


def multipart(*parts: bytes) -> io.BytesIO:
    body = b""
    for p in parts:
        body += (b"--myboundary\r\nContent-Type: image/jpeg\r\n"
                 + f"Content-Length: {len(p)}\r\n\r\n".encode() + p + b"\r\n")
    return io.BytesIO(body)


def test_iter_jpegs_splits_parts():
    assert list(iter_jpegs(multipart(b"\xff\xd8one", b"\xff\xd8two"))) == [
        b"\xff\xd8one", b"\xff\xd8two"]


def test_iter_jpegs_stops_on_truncated_part():
    stream = multipart(b"\xff\xd8one", b"\xff\xd8two")
    truncated = io.BytesIO(stream.getvalue()[:-6])
    assert list(iter_jpegs(truncated)) == [b"\xff\xd8one"]

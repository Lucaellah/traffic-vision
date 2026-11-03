import io
from datetime import datetime

from collector import config
from collector.official import parse_tdcs, parse_vd
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


VD_XML = b"""<?xml version='1.0' encoding='utf-8'?>
<VDLiveList xmlns="http://traffic.transportdata.tw/standard/traffic/schema/"><VDLives>
<VDLive><VDID>VD-N1-S-84.120-N-LOOP</VDID><LinkFlows><LinkFlow><LinkID>L1</LinkID><Lanes>
<Lane><LaneID>0</LaneID><LaneType>1</LaneType><Speed>38</Speed><Occupancy>46</Occupancy>
<Vehicles><Vehicle><VehicleType>S</VehicleType><Volume>25</Volume><Speed>37</Speed></Vehicle>
<Vehicle><VehicleType>L</VehicleType><Volume>1</Volume><Speed>30</Speed></Vehicle></Vehicles>
</Lane></Lanes></LinkFlow></LinkFlows><Status>0</Status>
<DataCollectTime>2026-11-03T11:09:00+08:00</DataCollectTime></VDLive>
<VDLive><VDID>VD-N1-N-69-O-SE-1</VDID><LinkFlows/><Status>0</Status></VDLive>
</VDLives></VDLiveList>"""


def test_parse_vd_keeps_only_wanted_detector():
    rows = parse_vd(VD_XML, ["VD-N1-S-84.120-N-LOOP"])
    assert rows == [
        ["2026-11-03T11:09:00+08:00", "VD-N1-S-84.120-N-LOOP", "L1", "0", "38", "46",
         "S", "25", "37", "0"],
        ["2026-11-03T11:09:00+08:00", "VD-N1-S-84.120-N-LOOP", "L1", "0", "38", "46",
         "L", "1", "30", "0"],
    ]


def test_parse_tdcs_section_needs_both_gantries():
    text = ("2026/11/03 10:00,01F0750S,01F0880S,31,500,100\n"
            "2026/11/03 10:00,01F0699S,01F0750S,31,223,85\n")
    assert len(parse_tdcs(text, "M04A", ["01F0750S", "01F0880S"])) == 1
    m03a = "2026-11-03 10:00,01F0750S,S,31,224\n2026-11-03 10:00,01F0017S,S,31,9\n"
    assert parse_tdcs(m03a, "M03A", ["01F0750S"]) == [["2026-11-03 10:00", "01F0750S", "S",
                                                       "31", "224"]]

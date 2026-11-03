"""下載一天份的官方交通資料，只留下湖口路段的設備，當作影像分析的驗證基準。

- VD（每分鐘）：各車道的速度、佔有率，以及小車 S / 大車 L / 聯結車 T 的流量
- ETC 門架（每 5 分鐘，TDCS）：M03A 各車種通過量、M04A 旅行時間、M05A 區間車速
  車種代碼：31 小客車、32 小貨車、41 大客車、42 大貨車、5 聯結車

範例：
    python -m collector.official                  # 下載昨天
    python -m collector.official --date 20261103  # 指定日期（當天的資料只到目前為止）
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import sys
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

from collector import config

NS = "{http://traffic.transportdata.tw/standard/traffic/schema/}"
USER_AGENT = "traffic-vision-collector/0.1 (personal research)"

VD_COLUMNS = ["collect_time", "vd_id", "link_id", "lane_id", "lane_speed", "occupancy",
              "vehicle_type", "volume", "speed", "status"]
TDCS_COLUMNS = {
    "M03A": ["time", "gantry", "direction", "vehicle_type", "volume"],
    "M04A": ["time", "gantry_from", "gantry_to", "vehicle_type", "travel_time_s", "volume"],
    "M05A": ["time", "gantry_from", "gantry_to", "vehicle_type", "speed", "volume"],
}


def fetch(url: str) -> bytes | None:
    """下載檔案；不存在（404）時回傳 None。"""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=30) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if attempt == 2:
                raise
        except (urllib.error.URLError, OSError):
            if attempt == 2:
                raise
    return None


def parse_vd(xml: bytes, vd_ids: Iterable[str]) -> list[list]:
    """從 VDLive XML 取出指定 VD 的資料，每個車道 × 車種一列。"""
    wanted = set(vd_ids)
    rows = []
    for vd in ET.fromstring(xml).iter(f"{NS}VDLive"):
        vd_id = vd.findtext(f"{NS}VDID")
        if vd_id not in wanted:
            continue
        collect = vd.findtext(f"{NS}DataCollectTime")
        status = vd.findtext(f"{NS}Status")
        for link in vd.iter(f"{NS}LinkFlow"):
            link_id = link.findtext(f"{NS}LinkID")
            for lane in link.iter(f"{NS}Lane"):
                base = [collect, vd_id, link_id, lane.findtext(f"{NS}LaneID"),
                        lane.findtext(f"{NS}Speed"), lane.findtext(f"{NS}Occupancy")]
                for v in lane.iter(f"{NS}Vehicle"):
                    rows.append(base + [v.findtext(f"{NS}VehicleType"),
                                        v.findtext(f"{NS}Volume"), v.findtext(f"{NS}Speed"),
                                        status])
    return rows


def parse_tdcs(text: str, kind: str, gantries: Iterable[str]) -> list[list[str]]:
    """M03A 留下指定門架；M04A / M05A 只留起訖門架都在清單裡的路段。"""
    wanted = set(gantries)
    rows = []
    for row in csv.reader(io.StringIO(text)):
        if not row:
            continue
        if kind == "M03A":
            keep = row[1] in wanted
        else:
            keep = row[1] in wanted and row[2] in wanted
        if keep:
            rows.append(row)
    return rows


def vd_day(day: str, workers: int) -> list[list]:
    base = f"{config.HISTORY_URL}/motc20/VD"
    urls = [f"{base}/{day}/VDLive_{h:02d}{m:02d}.xml.gz" for h in range(24) for m in range(60)]
    # 檔名是發布時間，裡面是前一分鐘的資料：當天 23:59 的資料在隔天的 0000 檔
    next_day = (datetime.strptime(day, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d")
    urls.append(f"{base}/{next_day}/VDLive_0000.xml.gz")
    date_prefix = f"{day[:4]}-{day[4:6]}-{day[6:]}"

    def one(url: str) -> list[list]:
        data = fetch(url)
        return parse_vd(gzip.decompress(data), config.VD_IDS) if data else []

    with ThreadPoolExecutor(workers) as pool:
        rows = [r for part in pool.map(one, urls) for r in part]
    # 同一分鐘的資料可能出現在相鄰兩個檔案裡；只留這一天的
    unique = {tuple(r[:4] + r[6:7]): r for r in rows if r[0].startswith(date_prefix)}
    return sorted(unique.values(), key=lambda r: (r[0], r[1], r[3], r[6]))


def tdcs_day(day: str, kind: str, workers: int) -> list[list[str]]:
    base = f"{config.HISTORY_URL}/TDCS/{kind}/{day}"
    urls = [f"{base}/{h:02d}/TDCS_{kind}_{day}_{h:02d}{m:02d}00.csv"
            for h in range(24) for m in range(0, 60, 5)]

    def one(url: str) -> list[list[str]]:
        data = fetch(url)
        return parse_tdcs(data.decode("utf-8-sig"), kind, config.GANTRIES) if data else []

    with ThreadPoolExecutor(workers) as pool:
        return [r for part in pool.map(one, urls) for r in part]


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="") as f:
        out = csv.writer(f)
        out.writerow(header)
        out.writerows(rows)


def download(day: str, workers: int) -> bool:
    out_dir = config.RAW_DIR / "official" / day
    vd = vd_day(day, workers)
    write_csv(out_dir / "vd.csv.gz", VD_COLUMNS, vd)
    counts = {"vd": len(vd)}
    for kind, header in TDCS_COLUMNS.items():
        rows = tdcs_day(day, kind, workers)
        write_csv(out_dir / f"{kind.lower()}.csv.gz", header, rows)
        counts[kind.lower()] = len(rows)
    print(f"{day} -> {out_dir}  " + "  ".join(f"{k}={v}" for k, v in counts.items()))
    return all(counts.values())


def main() -> int:
    parser = argparse.ArgumentParser(description="下載湖口路段的官方 VD 與 ETC 門架資料")
    parser.add_argument("--date", default=None,
                        help="YYYYMMDD；不指定時補抓最近幾天還沒下載的日期（不含今天）")
    parser.add_argument("--catch-up", type=int, default=3, help="不指定日期時往回檢查幾天")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    if args.date:
        datetime.strptime(args.date, "%Y%m%d")  # 格式錯誤時直接報錯
        days = [args.date]
    else:
        today = config.now()
        days = [(today - timedelta(days=n)).strftime("%Y%m%d")
                for n in range(args.catch_up, 0, -1)]
        days = [d for d in days if not (config.RAW_DIR / "official" / d / "vd.csv.gz").exists()]
    ok = [download(day, args.workers) for day in days]
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())

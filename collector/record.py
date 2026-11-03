"""錄一段高公局 CCTV 的 MJPEG 串流，存成 H.264 影片，並逐幀記錄時間。

串流大約每 9 秒會斷線一次，所以會自動重連；斷線期間沒有畫面，幀與幀的間隔並不均勻。
分析時應以同名的 .frames.csv 為準，不要用影片的 fps 推算時間。

範例：
    python -m collector.record                    # 錄 18460 鏡頭 60 秒
    python -m collector.record --scheduled        # 給 systemd timer 用：不在排程時段就直接結束
"""

from __future__ import annotations

import argparse
import csv
import http.client
import io
import json
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import BinaryIO

import cv2
import numpy as np
from PIL import Image

from collector import config
from traffic_vision.video import H264Writer

USER_AGENT = "traffic-vision-collector/0.1 (personal research)"
EXIF_DATETIME = 306


def iter_jpegs(fp: BinaryIO) -> Iterator[bytes]:
    """從 multipart/x-mixed-replace 串流逐張取出 JPEG。串流結束時停止。"""
    while True:
        line = fp.readline()
        if not line:
            return
        if not line.startswith(b"--"):
            continue
        length = None
        while True:
            header = fp.readline()
            if not header:
                return
            header = header.strip()
            if not header:
                break
            name, _, value = header.partition(b":")
            if name.strip().lower() == b"content-length":
                length = int(value.strip())
        if length is None:
            raise ValueError("串流的 part 沒有 Content-Length")
        data = fp.read(length)
        if len(data) < length:
            return
        yield data


def camera_time(jpeg: bytes) -> str:
    """攝影機寫在 EXIF 裡的拍攝時間（精確到秒），讀不到時回傳空字串。"""
    try:
        value = Image.open(io.BytesIO(jpeg)).getexif().get(EXIF_DATETIME, "")
    except Exception:
        return ""
    return str(value).replace(":", "-", 2)


def capture(url: str, seconds: float) -> tuple[list[tuple[float, bytes]], dict]:
    """在 seconds 秒內盡量收集畫面；斷線就重連。回傳 (接收時間, JPEG) 與連線統計。"""
    frames: list[tuple[float, bytes]] = []
    stats = {"connections": 0, "errors": 0, "duplicates": 0}
    deadline = time.time() + seconds
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    while time.time() < deadline:
        stats["connections"] += 1
        try:
            with urllib.request.urlopen(request, timeout=10) as resp:
                for jpeg in iter_jpegs(resp):
                    t = time.time()
                    if t >= deadline:
                        break
                    if frames and jpeg == frames[-1][1]:
                        stats["duplicates"] += 1
                        continue
                    frames.append((t, jpeg))
        except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError):
            stats["errors"] += 1
            time.sleep(1)
        else:
            time.sleep(0.2)
    return frames, stats


def clip_path(camera: str, slot: datetime) -> Path:
    day = slot.strftime("%Y%m%d")
    return config.RAW_DIR / "cctv" / camera / day / f"cam{camera}_{slot:%Y%m%d_%H%M}.mp4"


def save(frames: list[tuple[float, bytes]], dst: Path) -> dict:
    """把畫面編碼成 H.264，並寫出逐幀時間表。fps 以實際收到的幀數換算。"""
    images = [cv2.imdecode(np.frombuffer(j, np.uint8), cv2.IMREAD_COLOR) for _, j in frames]
    keep = [i for i, img in enumerate(images) if img is not None]
    frames = [frames[i] for i in keep]
    images = [images[i] for i in keep]
    times = [t for t, _ in frames]
    span = times[-1] - times[0]
    fps = (len(times) - 1) / span if span > 0 else 1.0
    h, w = images[0].shape[:2]

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".part.mp4")
    writer = H264Writer(tmp, w, h, fps)
    for img in images:
        writer.write(img)
    writer.close()
    tmp.replace(dst)

    with open(dst.with_suffix(".frames.csv"), "w", newline="") as f:
        out = csv.writer(f)
        out.writerow(["frame", "recv_time", "recv_ts", "camera_time", "gap_s"])
        for i, (t, jpeg) in enumerate(frames):
            gap = t - times[i - 1] if i else 0.0
            recv = datetime.fromtimestamp(t, config.TZ).isoformat(timespec="milliseconds")
            out.writerow([i, recv, f"{t:.3f}", camera_time(jpeg), f"{gap:.3f}"])

    gaps = np.diff(times) if len(times) > 1 else np.array([0.0])
    return {
        "frames": len(times),
        "fps": round(fps, 3),
        "size": f"{w}x{h}",
        "span_s": round(span, 2),
        "max_gap_s": round(float(gaps.max()), 2),
        # 間隔超過 2 秒視為斷線，這段時間通過的車不會被看到
        "covered_s": round(float(gaps[gaps <= 2].sum()), 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="錄一段高公局 CCTV 影片")
    parser.add_argument("--camera", default=config.CAMERA)
    parser.add_argument("--seconds", type=float, default=config.CLIP_SECONDS)
    parser.add_argument("--scheduled", action="store_true",
                        help="依 config 的排程判斷這個時段要不要錄，不用錄就直接結束")
    args = parser.parse_args()

    start = config.now()
    if args.scheduled and not config.should_record(start):
        return 0
    slot = start.replace(minute=start.minute // 5 * 5, second=0, microsecond=0)
    dst = clip_path(args.camera, slot)
    url = config.STREAM_URL.format(camera=args.camera)

    frames, stats = capture(url, args.seconds)
    meta = {
        "camera": args.camera,
        "stream_url": url,
        "slot": slot.isoformat(),
        "started_at": start.isoformat(timespec="seconds"),
        "requested_s": args.seconds,
        **stats,
    }
    if frames:
        meta.update(save(frames, dst))
    else:
        meta["frames"] = 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(json.dumps(meta, ensure_ascii=False))
    return 0 if frames else 1


if __name__ == "__main__":
    sys.exit(main())

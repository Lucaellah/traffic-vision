"""影片的多目標追蹤、軌跡繪製與越線計數。"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .counting import LineCounter, Point
from .detection import box_xywh, result_boxes

TRACK_COLUMNS = [
    "frame", "time_s", "track_id", "class_id", "class_name", "conf", "x", "y", "w", "h",
]

TRAIL_COLOR = (0, 170, 250)  # BGR
LINE_COLOR = (80, 220, 80)


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    frames: int

    @property
    def duration_s(self) -> float:
        return self.frames / self.fps if self.fps else 0.0


@dataclass
class VideoResult:
    output_path: Path
    tracks: pd.DataFrame
    counter: LineCounter | None
    info: VideoInfo
    processed_frames: int


def probe(path: str | Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"無法開啟影片：{path}")
    try:
        return VideoInfo(
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=cap.get(cv2.CAP_PROP_FPS) or 30.0,
            frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        cap.release()


def read_first_frame(path: str | Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    try:
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        raise ValueError(f"無法讀取影片畫面：{path}")
    return frame


class H264Writer:
    """用 imageio-ffmpeg 內建的 ffmpeg 輸出 H.264，瀏覽器才能直接播放（OpenCV 的 mp4v 不行）。"""

    def __init__(self, path: str | Path, width: int, height: int, fps: float):
        import imageio_ffmpeg

        # yuv420p 需要偶數寬高
        self.width, self.height = width // 2 * 2, height // 2 * 2
        self._gen = imageio_ffmpeg.write_frames(
            str(path),
            (self.width, self.height),
            fps=fps,
            codec="libx264",
            pix_fmt_in="bgr24",
            quality=7,
            macro_block_size=1,
            output_params=["-movflags", "+faststart"],
        )
        self._gen.send(None)

    def write(self, frame_bgr: np.ndarray) -> None:
        self._gen.send(np.ascontiguousarray(frame_bgr[: self.height, : self.width]))

    def close(self) -> None:
        self._gen.close()


def draw_line(canvas: np.ndarray, counter: LineCounter) -> None:
    p1 = tuple(int(v) for v in counter.start)
    p2 = tuple(int(v) for v in counter.end)
    cv2.line(canvas, p1, p2, LINE_COLOR, 3, cv2.LINE_AA)
    text = f"IN {counter.count_in}  OUT {counter.count_out}  TOTAL {counter.total}"
    scale = max(canvas.shape[1] / 1280, 0.6)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    cv2.rectangle(canvas, (16, 16), (32 + tw, 32 + th + 8), (0, 0, 0), -1)
    cv2.putText(canvas, text, (24, 24 + th), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), 2, cv2.LINE_AA)


def draw_trails(canvas: np.ndarray, trails: dict[int, deque], active_ids: list[int]) -> None:
    for tid in active_ids:
        pts = trails[tid]
        if len(pts) > 1:
            arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [arr], False, TRAIL_COLOR, 3, cv2.LINE_AA)


def process_video(
    src: str | Path,
    dst: str | Path,
    model,
    *,
    conf: float = 0.25,
    iou: float = 0.7,
    classes: list[int] | None = None,
    line: tuple[Point, Point] | None = None,
    max_frames: int | None = None,
    trail_length: int = 60,
    show_labels: bool = True,
    progress: Callable[[int, int], None] | None = None,
) -> VideoResult:
    """逐幀追蹤物件，輸出標註影片並回傳每幀每個物件的位置紀錄。

    model 必須是這次處理專用的 YOLO 實例：追蹤器狀態存在 model 上。
    """
    info = probe(src)
    total = min(info.frames, max_frames) if max_frames else info.frames
    counter = LineCounter(*line) if line else None
    trails: dict[int, deque] = defaultdict(lambda: deque(maxlen=trail_length))
    rows: list[list] = []

    cap = cv2.VideoCapture(str(src))
    writer: H264Writer | None = None
    idx = 0
    try:
        while not (max_frames and idx >= max_frames):
            ok, frame = cap.read()
            if not ok:
                break
            # 第一幀 persist=False 會重建追蹤器，確保不沿用上一支影片的 id
            result = model.track(frame, persist=idx > 0, conf=conf, iou=iou,
                                 classes=classes, verbose=False)[0]
            # 車多時標籤很容易蓋住畫面：縮小線寬、拿掉信心分數
            canvas = result.plot(conf=False, labels=show_labels, line_width=2)
            active: list[int] = []
            boxes = result_boxes(result)
            if boxes is not None and boxes.id is not None:
                for (x, y, w, h), tid, c, p in zip(
                    box_xywh(boxes),
                    boxes.id.int().cpu().tolist(),
                    boxes.cls.int().cpu().tolist(),
                    boxes.conf.cpu().numpy(),
                    strict=True,
                ):
                    x, y = float(x), float(y)
                    rows.append([idx, idx / info.fps, tid, c, result.names[c], float(p),
                                 x, y, float(w), float(h)])
                    trails[tid].append((x, y))
                    active.append(tid)
                    if counter:
                        counter.update(tid, (x, y))
            draw_trails(canvas, trails, active)
            if counter:
                draw_line(canvas, counter)

            if writer is None:
                writer = H264Writer(dst, canvas.shape[1], canvas.shape[0], info.fps)
            writer.write(canvas)
            idx += 1
            if progress:
                progress(idx, total)
    finally:
        cap.release()
        if writer:
            writer.close()

    if idx == 0:
        raise ValueError("影片中沒有可讀取的畫面")
    return VideoResult(Path(dst), pd.DataFrame(rows, columns=TRACK_COLUMNS), counter, info, idx)

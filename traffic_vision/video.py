"""影片的多目標追蹤、軌跡繪製與越線計數。"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .counting import LineCounter, Point
from .labels import category_ids
from .tracking import MotionTracker

TRACK_COLUMNS = [
    "frame", "time_s", "track_id", "class_id", "class_name", "conf", "x", "y", "w", "h",
]

CROSSING_COLUMNS = ["line", "track_id", "direction", "time_s"]

TRAIL_COLOR = (0, 170, 250)  # BGR
LINE_COLOR = (80, 220, 80)
SPEED_LINE_COLOR = (40, 160, 255)  # 橘色（BGR）


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
    # 越線紀錄：line 為 "count"（計數線）或 "speed"（測速線）
    crossings: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=CROSSING_COLUMNS))


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
    draw_in_arrow(canvas, counter)
    text = f"IN {counter.count_in}  OUT {counter.count_out}  TOTAL {counter.total}"
    scale = max(canvas.shape[1] / 1280, 0.6)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    cv2.rectangle(canvas, (16, 16), (32 + tw, 32 + th + 8), (0, 0, 0), -1)
    cv2.putText(canvas, text, (24, 24 + th), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), 2, cv2.LINE_AA)


def draw_speed_line(canvas: np.ndarray, counter: LineCounter) -> None:
    p1 = tuple(int(v) for v in counter.start)
    p2 = tuple(int(v) for v in counter.end)
    cv2.line(canvas, p1, p2, SPEED_LINE_COLOR, 2, cv2.LINE_AA)
    # 在線的右端上方標示，避免和計數線搞混
    scale = max(canvas.shape[1] / 1280, 0.4)
    (tw, th), _ = cv2.getTextSize("SPEED", cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    right = max(p1, p2)
    org = (min(right[0], canvas.shape[1] - tw - 4) - tw // 2, right[1] - 6)
    org = (max(org[0], 4), max(org[1], th + 4))
    cv2.putText(canvas, "SPEED", org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(canvas, "SPEED", org, cv2.FONT_HERSHEY_SIMPLEX, scale, SPEED_LINE_COLOR, 1,
                cv2.LINE_AA)


def draw_in_arrow(canvas: np.ndarray, counter: LineCounter) -> None:
    """在計數線中點畫一個穿過線的箭頭，箭頭方向就是 IN。"""
    normal = counter.in_normal()
    if normal is None:
        return
    nx, ny = normal
    mx = (counter.start[0] + counter.end[0]) / 2
    my = (counter.start[1] + counter.end[1]) / 2
    half = max(min(canvas.shape[:2]) * 0.06, 20)
    tail = (int(mx - nx * half), int(my - ny * half))
    head = (int(mx + nx * half), int(my + ny * half))
    cv2.arrowedLine(canvas, tail, head, (0, 0, 0), 9, cv2.LINE_AA, tipLength=0.4)  # 黑邊比較醒目
    cv2.arrowedLine(canvas, tail, head, LINE_COLOR, 4, cv2.LINE_AA, tipLength=0.4)

    scale = max(canvas.shape[1] / 1280, 0.6)  # 和左上角計數文字一樣大
    (tw, th), _ = cv2.getTextSize("IN", cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    # 文字放在箭頭尖端外側，以文字中心對齊
    cx = mx + nx * (half + th + 6)
    cy = my + ny * (half + th + 6)
    org = (int(cx - tw / 2), int(cy + th / 2))
    cv2.putText(canvas, "IN", org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.putText(canvas, "IN", org, cv2.FONT_HERSHEY_SIMPLEX, scale, LINE_COLOR, 2, cv2.LINE_AA)


def draw_trails(canvas: np.ndarray, trails: dict[int, deque], active_ids: list[int]) -> None:
    for tid in active_ids:
        pts = trails[tid]
        if len(pts) > 1:
            arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [arr], False, TRAIL_COLOR, 3, cv2.LINE_AA)


# 同一台車常同時被判成 car 和 truck（框幾乎完全重疊），只在這些類別之間去重，
# 騎士和機車的框本來就會重疊，不能一起合併
VEHICLE_CATEGORIES = ["汽車", "公車", "卡車", "三輪車", "其他車輛"]


def merge_vehicle_duplicates(result, iou: float = 0.7):
    """四輪車類別之間重疊超過 iou 的框只留信心最高的一個。"""
    from ultralytics.utils.metrics import box_iou

    boxes = result.boxes
    vehicle = np.isin(boxes.cls.int().cpu().numpy(),
                      category_ids(result.names, VEHICLE_CATEGORIES))
    if vehicle.sum() < 2:
        return result
    overlap = box_iou(boxes.xyxy, boxes.xyxy).cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    drop = np.zeros(len(boxes), dtype=bool)
    for i in np.argsort(-conf):
        if vehicle[i] and not drop[i]:
            drop |= vehicle & (overlap[i] > iou) & (np.arange(len(boxes)) != i)
    return result[np.flatnonzero(~drop)] if drop.any() else result


def plot_detections(result, tracks: np.ndarray, show_labels: bool) -> np.ndarray:
    """畫出所有偵測框，已接上軌跡的框用追蹤後的位置並標上 ID。

    新出現的物件要再被偵測到一次才會成為軌跡，只畫軌跡的話框會時有時無。
    """
    import torch

    tracked_idx = tracks[:, -1].astype(int)
    rest = np.setdiff1d(np.arange(len(result.boxes)), tracked_idx)
    # 車多時標籤很容易蓋住畫面：縮小線寬、拿掉信心分數；低解析度影片（如 CCTV）再縮小一點
    style = {"conf": False, "labels": show_labels,
             "line_width": 1 if result.orig_shape[1] < 640 else 2}
    canvas = result[rest].plot(**style)
    if len(tracks):
        tracked = result[tracked_idx]
        tracked.update(boxes=torch.as_tensor(tracks[:, :-1], device=result.boxes.data.device))
        canvas = tracked.plot(img=canvas, **style)
    return canvas


def process_video(
    src: str | Path,
    dst: str | Path,
    model,
    *,
    conf: float = 0.25,
    iou: float = 0.7,
    classes: list[int] | None = None,
    imgsz: int | None = None,
    line: tuple[Point, Point] | None = None,
    speed_line: tuple[Point, Point] | None = None,
    max_frames: int | None = None,
    trail_length: int = 60,
    show_labels: bool = True,
    progress: Callable[[int, int], None] | None = None,
) -> VideoResult:
    """逐幀追蹤物件，輸出標註影片並回傳每幀每個物件的位置紀錄。

    imgsz 為 None 時沿用模型訓練時的解析度。
    畫面上會畫出每一幀的所有偵測框；已經接上軌跡的框另外標上 ID。
    speed_line 是測速用的第二條線：車輛通過兩條線的時間差搭配實際距離就能算出車速。
    """
    size = {"imgsz": imgsz} if imgsz else {}
    info = probe(src)
    total = min(info.frames, max_frames) if max_frames else info.frames
    tracker = MotionTracker(info.fps)
    counter = LineCounter(*line) if line else None
    speed_counter = LineCounter(*speed_line) if speed_line else None
    lines = [c for c in (counter, speed_counter) if c]
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
            result = model.predict(frame, conf=conf, iou=iou, classes=classes,
                                   verbose=False, **size)[0]
            result = merge_vehicle_duplicates(result)
            tracks = tracker.update(result.boxes.cpu().numpy(), frame)
            canvas = plot_detections(result, tracks, show_labels)
            active: list[int] = []
            for x1, y1, x2, y2, tid, p, c, _ in tracks:
                tid, c = int(tid), int(c)
                x, y, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
                rows.append([idx, idx / info.fps, tid, c, result.names[c], float(p),
                             float(x), float(y), float(w), float(h)])
                trails[tid].append((x, y))
                active.append(tid)
                for c in lines:
                    c.update(tid, (x, y), idx / info.fps)
            for tid, (x1, y1, x2, y2) in tracker.pending():
                for c in lines:
                    c.update(tid, ((x1 + x2) / 2, (y1 + y2) / 2), idx / info.fps)
            draw_trails(canvas, trails, active)
            if speed_counter:
                draw_speed_line(canvas, speed_counter)
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
    crossings = pd.DataFrame(
        [(name, c.track_id, c.direction, c.time_s)
         for name, lc in (("count", counter), ("speed", speed_counter)) if lc
         for c in lc.crossings],
        columns=CROSSING_COLUMNS,
    )
    return VideoResult(Path(dst), pd.DataFrame(rows, columns=TRACK_COLUMNS), counter, info, idx,
                       crossings)

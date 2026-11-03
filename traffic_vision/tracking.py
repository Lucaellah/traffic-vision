"""能處理低幀率影片的多目標追蹤器。

Ultralytics 內建的追蹤器都用前後幀框的 IoU 配對，卡爾曼濾波器的參數也是照 30 fps 設的。
國道 CCTV 這類約 5 fps 的影片，車子一幀就移動將近一個車身：前後框幾乎不重疊，
卡爾曼預測的位置也一直落後，軌跡接不起來或在車與車之間跳來跳去。

這裡沿用 ByteTrack 的流程（高低信心兩階段配對、未確認軌跡、遺失後找回），
但配對改用「預測位置與偵測框中心的距離」：
速度直接由最近兩次實際觀測估計，距離以框的大小為單位，所以和幀率、物體遠近無關。
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from ultralytics.trackers.byte_tracker import BYTETracker, STrack, TrackState
from ultralytics.trackers.utils import matching

# 和 Ultralytics bytetrack.yaml 相同（track_buffer 另外依幀率設定）
BYTETRACK_ARGS = dict(
    track_high_thresh=0.25,
    track_low_thresh=0.1,
    new_track_thresh=0.25,
    match_thresh=0.8,
    fuse_score=False,
)

# 允許的位移誤差（以框的大小為單位）。剛出現的軌跡還不知道速度，要放寬
GATE_WITH_VELOCITY = 1.0
GATE_NO_VELOCITY = 2.0
MAX_AREA_RATIO = 3.0  # 面積差太多就不是同一個物體


class MotionTrack(STrack):
    """記錄每次實際觀測到的框，用來估計速度；輸出位置也直接用觀測值，不用卡爾曼平滑後的值。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observations: list[tuple[int, np.ndarray]] = []

    def activate(self, kalman_filter, frame_id: int):
        super().activate(kalman_filter, frame_id)
        self.observations = [(frame_id, self._tlwh.copy())]

    def update(self, new_track, frame_id: int):
        super().update(new_track, frame_id)
        self.observations = [*self.observations[-1:], (frame_id, new_track.tlwh)]

    def re_activate(self, new_track, frame_id: int, new_id: bool = False):
        super().re_activate(new_track, frame_id, new_id)
        self.observations = [*self.observations[-1:], (frame_id, new_track.tlwh)]

    @property
    def tlwh(self) -> np.ndarray:
        if getattr(self, "observations", None):
            return self.observations[-1][1].copy()
        return super().tlwh

    def predict_center(self, frame_id: int) -> tuple[np.ndarray, bool]:
        """預測在 frame_id 的框中心；第二個值表示是否已經知道速度。"""
        (f1, b1) = self.observations[-1]
        c1 = b1[:2] + b1[2:] / 2
        if len(self.observations) < 2:
            return c1, False
        (f0, b0) = self.observations[-2]
        velocity = (c1 - (b0[:2] + b0[2:] / 2)) / max(f1 - f0, 1)
        return c1 + velocity * (frame_id - f1), True


def motion_distance(tracks: list[MotionTrack], dets: list[STrack], frame_id: int) -> np.ndarray:
    """0～1 的配對成本：預測中心到偵測中心的距離 ÷ 框大小 ÷ 容許範圍。

    超出容許範圍或面積差太多時成本為 1（不配對）。
    """
    cost = np.ones((len(tracks), len(dets)), dtype=np.float32)
    if not tracks or not dets:
        return cost
    det_boxes = np.array([d.tlwh for d in dets], dtype=np.float32)
    det_centers = det_boxes[:, :2] + det_boxes[:, 2:] / 2
    det_areas = det_boxes[:, 2] * det_boxes[:, 3]
    for i, track in enumerate(tracks):
        center, has_velocity = track.predict_center(frame_id)
        box = track.observations[-1][1]
        area = box[2] * box[3]
        scale = np.sqrt(np.maximum(area, det_areas))
        gate = GATE_WITH_VELOCITY if has_velocity else GATE_NO_VELOCITY
        dist = np.linalg.norm(det_centers - center, axis=1) / np.maximum(scale, 1e-6) / gate
        ratio = np.maximum(area, det_areas) / np.maximum(np.minimum(area, det_areas), 1e-6)
        cost[i] = np.where(ratio > MAX_AREA_RATIO, 1.0, np.minimum(dist, 1.0))
    return cost


class MotionTracker(BYTETracker):
    """ByteTrack 流程，配對成本改用 motion_distance。"""

    track_class = MotionTrack

    def __init__(self, fps: float, **overrides):
        # 消失的軌跡保留約 1 秒（30 fps 時就是預設的 30 幀）
        args = BYTETRACK_ARGS | {"track_buffer": max(5, round(fps))} | overrides
        super().__init__(SimpleNamespace(**args))

    def get_dists(self, tracks, detections):
        return motion_distance(tracks, detections, self.frame_id)

    def _second_association(self, strack_pool, u_track, detections_second, activated, refind, lost):
        r_tracked = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        if r_tracked and detections_second:
            dists = motion_distance(r_tracked, detections_second, self.frame_id)
            matches, u_track, _ = matching.linear_assignment(dists, thresh=0.5)
            self._apply_matches(matches, r_tracked, detections_second, activated, refind)
        else:
            u_track = range(len(r_tracked))
        for i in u_track:
            r_tracked[i].mark_lost()
            lost.append(r_tracked[i])

    def update(self, results, img=None, **kwargs) -> np.ndarray:
        """回傳 (N, 8)：x1, y1, x2, y2, track_id, score, cls, 偵測索引。"""
        return super().update(results, img, **kwargs).reshape(-1, 8)

    def pending(self) -> list[tuple[int, np.ndarray]]:
        """這一幀剛出現、還沒確認的軌跡 (track_id, xyxy)。

        新軌跡要再被偵測到一次才會出現在 update() 的結果裡。低幀率時，
        車子可能在這一幀之間就已經越線，計數時需要它第一次出現的位置。
        """
        return [(t.track_id, t.xyxy) for t in self.tracked_stracks
                if not t.is_activated and t.frame_id == self.frame_id]

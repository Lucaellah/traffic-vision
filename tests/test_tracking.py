from types import SimpleNamespace

import numpy as np

from traffic_vision.tracking import MotionTracker


def dets(*boxes):
    """模擬 Ultralytics Boxes.cpu().numpy()：只需要 conf、cls、xywh、xyxy 和布林索引。"""
    xyxy = np.array(boxes, dtype=np.float32).reshape(-1, 4)

    class Dets(SimpleNamespace):
        def __len__(self):
            return len(self.conf)

        def __getitem__(self, mask):
            return Dets(xyxy=self.xyxy[mask], xywh=self.xywh[mask],
                        conf=self.conf[mask], cls=self.cls[mask])

    xywh = np.concatenate([(xyxy[:, :2] + xyxy[:, 2:]) / 2, xyxy[:, 2:] - xyxy[:, :2]], axis=1)
    return Dets(xyxy=xyxy, xywh=xywh, conf=np.full(len(xyxy), 0.9, np.float32),
                cls=np.full(len(xyxy), 2.0, np.float32))


def run(tracker, positions):
    ids = []
    for x, y in positions:
        out = tracker.update(dets([x, y, x + 10, y + 10]))
        ids.append(out[:, 4].astype(int).tolist())
    return ids


def test_low_fps_motion_keeps_one_id():
    """每幀往右移動 1.2 個車身，前後框完全不重疊，仍然是同一條軌跡。"""
    ids = run(MotionTracker(fps=5), [(12 * i, 50) for i in range(6)])
    assert ids[0] == [1]  # 第一幀直接確認
    assert sum(ids, []) and set(sum(ids, [])) == {1}
    assert all(ids[1:])


def test_far_jump_starts_new_track():
    ids = run(MotionTracker(fps=5), [(0, 50), (12, 50), (24, 50), (200, 50), (212, 50)])
    assert set(sum(ids[:3], [])) == {1}
    assert 1 not in sum(ids[3:], [])


def test_pending_reports_unconfirmed_first_position():
    tracker = MotionTracker(fps=5)
    tracker.update(dets([0, 0, 10, 10]))  # 第一幀會直接確認
    tracker.update(dets([0, 0, 10, 10], [100, 100, 110, 110]))
    pending = tracker.pending()
    assert len(pending) == 1
    np.testing.assert_allclose(pending[0][1], [100, 100, 110, 110])


def test_merge_vehicle_duplicates_keeps_rider_and_motorcycle():
    import torch
    from ultralytics.engine.results import Results

    from traffic_vision.video import merge_vehicle_duplicates

    names = {0: "person", 2: "car", 3: "motorcycle", 7: "truck"}
    boxes = torch.tensor([
        [0, 0, 10, 10, 0.6, 2],     # car
        [0, 0, 10, 10.2, 0.4, 7],   # 同一台車被判成 truck
        [50, 50, 60, 60, 0.8, 0],   # 騎士
        [50, 50, 60, 61, 0.7, 3],   # 機車
    ])
    result = Results(np.zeros((100, 100, 3), np.uint8), path="", names=names, boxes=boxes)
    kept = merge_vehicle_duplicates(result).boxes.cls.int().tolist()
    assert sorted(kept) == [0, 2, 3]

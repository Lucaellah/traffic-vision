import numpy as np
import pytest

from training.visdrone_eval import official_ap, outside_ignored, voc_ap

REGION = np.array([[0, 0, 50, 50]])  # 忽略區域：xywh


def test_outside_ignored():
    boxes = np.array([
        [10, 10, 30, 30],  # 完全在忽略區域內
        [60, 60, 80, 80],  # 完全在外面
        [40, 10, 70, 30],  # 約三分之一在內：保留
        [20, 10, 60, 30],  # 約四分之三在內：排除
    ])
    assert outside_ignored(boxes, REGION, 100, 100).tolist() == [False, True, True, False]


def test_outside_ignored_without_regions():
    boxes = np.array([[10, 10, 30, 30]])
    assert outside_ignored(boxes, np.zeros((0, 4)), 100, 100).tolist() == [True]


def test_voc_ap_perfect():
    assert voc_ap(np.array([0.5, 1.0]), np.array([1.0, 1.0])) == 1.0


def image(pred_conf):
    return {"gt_boxes": np.array([[0.0, 0, 10, 10]]), "gt_cls": np.array([0.0]),
            "pred_boxes": np.array([[0.0, 0, 10, 10], [50.0, 50, 60, 60]]),
            "pred_conf": np.array(pred_conf), "pred_cls": np.array([0.0, 0.0])}


def test_official_ap_ranks_by_confidence():
    # 正確的框分數較高：AP 為 1；誤報分數較高：精確率只剩一半
    assert official_ap([image([0.9, 0.1])], 1)[0, 0] == pytest.approx(1.0)
    assert official_ap([image([0.1, 0.9])], 1)[0, 0] == pytest.approx(0.5)

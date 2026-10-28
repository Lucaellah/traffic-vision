"""單張影像的物件偵測 / 實例分割。"""

from __future__ import annotations

import numpy as np
import pandas as pd

DETECTION_COLUMNS = ["class_id", "class_name", "conf", "x", "y", "w", "h"]


def result_boxes(result):
    """一般模型的結果在 .boxes，OBB（旋轉框）模型在 .obb。"""
    return result.boxes if result.boxes is not None else result.obb


def box_xywh(boxes) -> np.ndarray:
    """框的中心點與寬高；OBB 取 xywhr 的前四欄（忽略旋轉角）。"""
    xywh = boxes.xywh if hasattr(boxes, "xywh") else boxes.xywhr[:, :4]
    return xywh.cpu().numpy()


def detect_image(
    model,
    image_bgr: np.ndarray,
    *,
    conf: float = 0.25,
    iou: float = 0.7,
    classes: list[int] | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """回傳（標註後的 RGB 影像, 偵測結果表）。x、y 為框的中心點。"""
    result = model.predict(image_bgr, conf=conf, iou=iou, classes=classes, verbose=False)[0]
    annotated = result.plot()[:, :, ::-1]

    rows = []
    boxes = result_boxes(result)
    if boxes is not None and len(boxes):
        for (x, y, w, h), c, p in zip(
            box_xywh(boxes),
            boxes.cls.int().cpu().tolist(),
            boxes.conf.cpu().numpy(),
            strict=True,
        ):
            rows.append([c, result.names[c], float(p), float(x), float(y), float(w), float(h)])
    return annotated, pd.DataFrame(rows, columns=DETECTION_COLUMNS)

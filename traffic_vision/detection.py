"""單張影像的物件偵測 / 實例分割。"""

from __future__ import annotations

import numpy as np
import pandas as pd

DETECTION_COLUMNS = ["class_id", "class_name", "conf", "x", "y", "w", "h"]


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
    boxes = result.boxes
    if boxes is not None and len(boxes):
        for (x, y, w, h), c, p in zip(
            boxes.xywh.cpu().numpy(),
            boxes.cls.int().cpu().tolist(),
            boxes.conf.cpu().numpy(),
            strict=True,
        ):
            rows.append([c, result.names[c], float(p), float(x), float(y), float(w), float(h)])
    return annotated, pd.DataFrame(rows, columns=DETECTION_COLUMNS)

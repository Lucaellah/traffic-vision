# 零樣本與微調比較（VisDrone test）

共同類別：person、bicycle、car、motorcycle、bus、truck；輸入解析度 1024；Ultralytics `model.val()`。
行人 + 人（非行走）合併為 person，汽車 + 廂型車合併為 car，三輪車兩類不計。

| 模型 | 類型 | Precision | Recall | mAP50 | mAP50-95 |
| --- | --- | --- | --- | --- | --- |
| yolo26s | COCO 預訓練（零樣本） | 0.371 | 0.236 | 0.206 | 0.117 |
| visdrone-yolo26s | VisDrone 微調 | 0.618 | 0.472 | 0.487 | 0.284 |

各類別 AP50：

| 模型 | person | bicycle | car | motorcycle | bus | truck |
| --- | --- | --- | --- | --- | --- | --- |
| yolo26s | 0.219 | 0.073 | 0.526 | 0.093 | 0.196 | 0.130 |
| visdrone-yolo26s | 0.401 | 0.190 | 0.839 | 0.415 | 0.619 | 0.457 |

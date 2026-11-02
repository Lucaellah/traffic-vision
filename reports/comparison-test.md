# 模型比較

資料集：visdrone（test），輸入解析度 1024，Ultralytics `model.val()`。粗體為最佳。

| | visdrone-yolo26s | visdrone-yolo26s-obb |
| --- | --- | --- |
| mAP50 | 0.404 | **0.408** |
| mAP50-95 | 0.235 | **0.237** |
| Precision | **0.535** | 0.520 |
| Recall | 0.414 | **0.429** |
| 速度（ms / 張） | **2.75** | 2.76 |

各類別 AP50：

| | visdrone-yolo26s | visdrone-yolo26s-obb |
| --- | --- | --- |
| pedestrian | **0.412** | 0.409 |
| people | **0.236** | 0.232 |
| bicycle | **0.190** | 0.177 |
| car | **0.799** | 0.794 |
| van | 0.421 | **0.432** |
| truck | 0.457 | **0.477** |
| tricycle | 0.257 | **0.291** |
| awning-tricycle | **0.238** | 0.235 |
| bus | **0.619** | 0.605 |
| motor | 0.415 | **0.424** |

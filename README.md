# 🚦 Traffic Vision

在本機執行的交通影像分析 Web 工具：上傳圖片或影片，就能用 **YOLO26** 做物件偵測、多目標追蹤、軌跡繪製和雙向越線計數，並把結果整理成可以下載、可以再分析的軌跡資料集。

> **所有資料都在你的電腦上處理。** 上傳的檔案只存在本機暫存資料夾，不會送到任何外部服務。App 也關閉了 Ultralytics 預設開啟的匿名使用統計。

## 功能

| 頁面 | 功能 |
| --- | --- |
| 🖼️ 影像偵測 | 單張圖片的物件偵測或實例分割；可調整信心門檻、IoU、推論解析度、要偵測的類別；匯出偵測明細（CSV / Excel） |
| 🎞️ 影片追蹤與計數 | ByteTrack 多目標追蹤與軌跡繪製；計數線可設成任意角度，分別統計 IN / OUT；輸出可以直接在瀏覽器播放的 H.264 標註影片和逐幀軌跡資料 |
| 📊 資料分析 | 各類別物件數、每秒車流量、位置熱力圖、主要軌跡疊圖、單一物件軌跡與移動距離 / 速度 |

## 快速開始

```bash
python -m venv .venv && source .venv/bin/activate

# 有 NVIDIA GPU：依驅動版本安裝對應的 PyTorch（例如 CUDA 12.4）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
streamlit run app.py
```

用瀏覽器開啟 <http://localhost:8501>。第一次使用某個模型時，會自動把權重下載到 `weights/`。

**怎麼選模型**：選單預設是 **YOLO26s（平衡）**，適合路口監視器、行車記錄器這類平視畫面（COCO 類別）。無人機空拍、俯視畫面請選 **YOLO26s-visdrone（空拍 / 俯視）**，這是在 VisDrone 上微調的模型（見下方「微調車輛偵測模型」，需先訓練）。一般模型幾乎認不出從空中拍的車和機車。機車很多的平視街景可以試 **YOLO26s-idd（平視 / 行車）**，這是在印度道路資料集 IDD 上微調的模型，會分出騎士、電動三輪車（同樣需先訓練，見下方「平視版：IDD」）。

**推論解析度**：側邊欄可以選 640、1024、1280、1536，預設「自動」沿用模型訓練時的設定（一般模型 640、空拍模型 1536）。解析度越大越能抓到遠處的小物體，但速度越慢。空拍模型以 1536 訓練，VisDrone 測試集 mAP50（官方規則）0.471；改用 1024 推論快約 2 倍，但會掉到 0.417。比訓練解析度大太多反而會變差，因為物體會比訓練時看過的還大。

### Docker（CPU）

```bash
docker build -t traffic-vision .
docker run --rm -p 8501:8501 traffic-vision
```

## 微調車輛偵測模型

官方的 YOLO 權重是用 COCO（日常照片）訓練的，對空拍、俯視畫面裡的小車效果有限，也分不出廂型車、三輪車。`training/` 提供完整的微調流程，預設使用 [VisDrone2019-DET](https://github.com/VisDrone/VisDrone-Dataset)（無人機視角，10 個類別，約 2 GB）：

```bash
# 1. 下載並轉換資料集到 datasets/（只需執行一次，訓練時也會自動執行）
python -m training.datasets visdrone

# 2. 微調：預設 YOLO26s、imgsz 1024、100 epochs，驗證集 20 個 epoch 沒進步就提早停止
python -m training.finetune

# 3. 用 Ultralytics 官方驗證器在測試集（test-dev）評估，報告輸出到 reports/
python -m training.evaluate --weights weights/custom/visdrone-yolo26s.pt --split test

# 4. 跨資料集比較：預訓練與微調模型分別在 COCO val2017、VisDrone test-dev（和 IDD）上評估
#    （第一次會下載 COCO val2017，約 1 GB）
python -m training.crossdomain

# 5. 依 VisDrone 官方評估工具的規則重算（處理忽略區域、官方計分方式）
python -m training.visdrone_eval --weights weights/custom/visdrone-yolo26s.pt --split test
```

平視版改用 IDD（見下方「平視版：IDD」）：

```bash
python -m training.idd                 # 轉換已下載的 IDD Detection（放在 datasets/IDD/）
python -m training.finetune --data idd
python -m training.evaluate --weights weights/custom/idd-yolo26s.pt --data idd --split test
python -m training.crossdomain --finetuned experiments/visdrone-yolo26s-1024.pt --idd custom/idd-yolo26s.pt
```

想試其他大小的模型時，用 `--model yolo26m.pt` 再訓練一次，評估後以 `python -m training.compare` 把多份報告整理成比較表。

**結果**：YOLO26s 在 VisDrone 上微調（imgsz 1024、batch 16，第 60 個 epoch 提早停止，RTX 4090 約 64 分鐘）。驗證集（val）只用來挑選最佳 epoch，以下 VisDrone 的分數全部來自訓練過程完全沒用到的測試集 test-dev（1,610 張），用官方驗證器評估。

10 個 VisDrone 類別：mAP50 **0.404**、mAP50-95 **0.235**、Precision 0.535、Recall 0.414，每張圖 2.75 ms。

**跨資料集比較**：COCO 預訓練、VisDrone 微調和 IDD 微調的 YOLO26s，分別在 COCO val2017（5,000 張）、VisDrone test-dev、IDD 測試集上評估。三個資料集的類別不同，只比共同的 6 類：VisDrone 的行人和「人（非行走）」合併為「人」，汽車和廂型車合併為「汽車」，三輪車兩類不計；IDD 的行人和騎士合併為「人」（COCO 的 person 包含騎車的人），電動三輪車、其他車輛等不計。評估解析度用各資料集的標準設定（COCO 640、VisDrone 和 IDD 1024），VisDrone 微調模型在 COCO 上另外補測 1024。這裡的 VisDrone 微調模型是 1024 訓練的版本，讓三個模型的訓練解析度一致。

mAP50：

| 測試集 | 模型 | 人 | 腳踏車 | 汽車 | 機車 | 公車 | 卡車 | **6 類平均** | 評估解析度 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| COCO | COCO 預訓練 | 0.824 | 0.593 | 0.696 | 0.756 | 0.850 | 0.595 | **0.719** | 640 |
| COCO | VisDrone 微調 | 0.140 | 0.057 | 0.220 | 0.024 | 0.041 | 0.074 | **0.093** | 640 |
| COCO | VisDrone 微調 | 0.163 | 0.040 | 0.220 | 0.028 | 0.089 | 0.076 | **0.103** | 1024 |
| COCO | IDD 微調 | 0.480 | 0.265 | 0.506 | 0.374 | 0.435 | 0.073 | **0.356** | 640 |
| VisDrone | COCO 預訓練 | 0.219 | 0.073 | 0.526 | 0.093 | 0.196 | 0.130 | **0.206** | 1024 |
| VisDrone | VisDrone 微調 | 0.401 | 0.190 | 0.839 | 0.415 | 0.619 | 0.457 | **0.487** | 1024 |
| VisDrone | IDD 微調 | 0.100 | 0.016 | 0.067 | 0.016 | 0.000 | 0.003 | **0.034** | 1024 |
| IDD | COCO 預訓練 | 0.534 | 0.261 | 0.670 | 0.530 | 0.534 | 0.337 | **0.478** | 1024 |
| IDD | VisDrone 微調 | 0.240 | 0.020 | 0.398 | 0.284 | 0.027 | 0.075 | **0.174** | 1024 |
| IDD | IDD 微調 | 0.693 | 0.554 | 0.765 | 0.766 | 0.776 | 0.718 | **0.712** | 1024 |

mAP50-95：

| 測試集 | 模型 | 人 | 腳踏車 | 汽車 | 機車 | 公車 | 卡車 | **6 類平均** | 評估解析度 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| COCO | COCO 預訓練 | 0.596 | 0.370 | 0.469 | 0.509 | 0.722 | 0.425 | **0.515** | 640 |
| COCO | VisDrone 微調 | 0.061 | 0.030 | 0.126 | 0.012 | 0.031 | 0.035 | **0.049** | 640 |
| COCO | VisDrone 微調 | 0.069 | 0.021 | 0.129 | 0.012 | 0.055 | 0.035 | **0.053** | 1024 |
| COCO | IDD 微調 | 0.282 | 0.131 | 0.324 | 0.207 | 0.364 | 0.046 | **0.226** | 640 |
| VisDrone | COCO 預訓練 | 0.092 | 0.030 | 0.323 | 0.032 | 0.142 | 0.083 | **0.117** | 1024 |
| VisDrone | VisDrone 微調 | 0.163 | 0.082 | 0.543 | 0.177 | 0.439 | 0.300 | **0.284** | 1024 |
| VisDrone | IDD 微調 | 0.043 | 0.007 | 0.034 | 0.005 | 0.000 | 0.001 | **0.015** | 1024 |
| IDD | COCO 預訓練 | 0.302 | 0.147 | 0.461 | 0.269 | 0.394 | 0.239 | **0.302** | 1024 |
| IDD | VisDrone 微調 | 0.098 | 0.009 | 0.252 | 0.113 | 0.015 | 0.040 | **0.088** | 1024 |
| IDD | IDD 微調 | 0.406 | 0.344 | 0.541 | 0.473 | 0.593 | 0.525 | **0.480** | 1024 |

- **微調讓模型適應空拍視角**：VisDrone 上 6 類 mAP50 從 0.206 提升到 0.487；機車從 0.093 提升到 0.415，約 4.5 倍。
- **代價是忘掉平視視角**：同一個微調模型在 COCO 上從 0.719 掉到 0.093。提高評估解析度到 1024 幾乎沒有幫助（0.103），問題在視角和物體大小，不在解析度。這就是「災難性遺忘」。
- **IDD 也一樣**：IDD 微調模型在 IDD 上從 0.478 提升到 0.712，但在 COCO 上掉到 0.356、在 VisDrone 上只剩 0.034。平視資料集之間的遺忘比「平視 ↔ 空拍」輕得多，但仍然明顯。
- **實際使用的建議**：無人機、高處俯拍的畫面用 YOLO26s-visdrone；機車多的平視街景可以用 YOLO26s-idd；其他平視畫面用 YOLO26s 等一般模型。
- 評估流程檢查：同樣的流程下，預訓練模型在 COCO 完整 80 類的 mAP50-95 為 0.480，和官方公布的 48.6 相差不到 1 點（官方使用 pycocotools 計分，算法略有不同）。
- VisDrone 測試集的分數明顯低於驗證集（10 類 mAP50：test 0.404、val 0.499），這是 VisDrone 常見的現象，test-dev 的場景比較難。

完整數字見 [reports/visdrone-yolo26s-test.md](reports/visdrone-yolo26s-test.md)、[reports/crossdomain.md](reports/crossdomain.md)；驗證集的結果在 [reports/visdrone-yolo26s-val.md](reports/visdrone-yolo26s-val.md)。

> 註：YOLO26 預設評估時使用一對多偵測頭加 NMS；改用免 NMS 的端到端模式（`nms=False`）時，測試集 mAP50 為 0.394。

**依 VisDrone 官方規則重算**：Ultralytics 的資料集轉換腳本會直接刪掉「忽略區域」（類別 0）和 others（類別 11），所以上面的分數裡，落在忽略區域內的預測都被算成誤報。`training.visdrone_eval` 依照 [VisDrone 官方評估工具](https://github.com/VisDrone/VisDrone2018-DET-toolkit)的規則重算：

- 面積一半以上落在忽略區域內的標註和預測都不計分（同 `dropObjectsInIgr.m`）。test-dev 有 2,180 個忽略區域，排除了 414 個標註（0.6%）和約 4.4% 的預測。
- others 不屬於 10 個評估類別，官方工具實際上也不會計到它；截斷程度不影響計分，標註裡也沒有截斷超過 50% 的物件。
- 官方計分方式：每張圖最多 500 個框，信心分數高的先配對，以 VOC 方式計算 AP。

| 模型 / 推論解析度 | Ultralytics 計分 | + 忽略區域 | 官方計分 + 忽略區域 |
| --- | --- | --- | --- |
| YOLO26s-visdrone / 1024 | 0.405 | 0.412 | **0.412** |
| YOLO26s-visdrone / 1536 | 0.440 | 0.447 | **0.447** |
| OBB 起點 / 1024 | 0.408 | 0.415 | **0.415** |
| 1536 重訓 / 1024 | 0.411 | 0.418 | **0.417** |
| 1536 重訓 / 1536 | 0.462 | 0.471 | **0.471** |

（test-dev，10 類 mAP50。三種分數來自同一組預測；Ultralytics 計分這欄改成每張圖最多 500 個框，所以和上面的 0.404 差 0.001。）

- 忽略區域讓 mAP50 提高約 0.007（mAP50-95 約 0.003），提高最多的是機車（+0.021）、行人（+0.011）、腳踏車（+0.010）、人（+0.009）這些小而密集的類別，汽車、公車幾乎不變。
- 兩種計分方式的結果幾乎相同（差距 ≤ 0.001），真正有影響的只有忽略區域。
- 所有模型提高的幅度都差不多，所以本 README 其他比較的結論不變；要和論文或競賽的數字比較，請用最後一欄。
- 推論解析度提高到 1536 不用重新訓練就能多 0.035，原因是 VisDrone 原圖寬 1360～2000 px，用 1024 推論等於先把圖縮小，小物體受影響最大。
- 再用 1536 重新訓練（`python -m training.finetune --imgsz 1536 --batch 8 --name visdrone-yolo26s-1536`，第 41 個 epoch 最好，約 2.5 小時）：用 1536 推論時再多 0.024（mAP50-95 0.259 → 0.275），行人 +0.026、人 +0.046、腳踏車 +0.032、機車 +0.030 提高最多；用 1024 推論則只多 0.005，在誤差範圍內。也就是說，訓練和推論的解析度要一起提高才有效。完整數字見 [reports/visdrone-yolo26s-1536-test-official-1536.md](reports/visdrone-yolo26s-1536-test-official-1536.md)。
- App 的空拍模型（`weights/custom/visdrone-yolo26s.pt`）已換成這個 1536 版本；原本的 1024 版本移到 `weights/experiments/visdrone-yolo26s-1024.pt`。上面其他比較用的都是 1024 版本。

完整數字見 [reports/visdrone-yolo26s-test-official.md](reports/visdrone-yolo26s-test-official.md)。

**平視版：IDD**：[IDD（India Driving Dataset）](https://idd.insaan.iiit.ac.in/)是印度道路的行車畫面，機車、騎士、電動三輪車很多，車流型態比 KITTI、BDD100K 更接近台灣；KITTI 沒有機車類別，所以不考慮。IDD 需要先註冊才能下載，把 `idd-detection.tar.gz` 解壓到 `datasets/IDD/` 後執行 `python -m training.idd` 轉換。

- **切分**：官方 test 沒有公開標註，所以拿官方 val（10,225 張）當測試集；從官方 train 依影片序列切出約 10% 當驗證集（3,074 張），其餘 25,356 張訓練。IDD 是從連續影片取出的畫面，同一段序列只放在一個切分裡；官方 train 和 val 有 5 段序列重疊（3,139 張），已從訓練集移除。
- **類別**：12 類（人、騎士、腳踏車、機車、汽車、電動三輪車、公車、卡車、其他車輛、動物、紅綠燈、交通標誌）；caravan、trailer 太少，併入其他車輛，train 只有約 60 個框，捨棄。
- **訓練**：從 COCO 權重開始，imgsz 1024、batch 16，第 61 個 epoch 最好、第 81 個 epoch 提早停止，RTX 4090 約 4.6 小時。

測試集 12 類 mAP50 **0.587**、mAP50-95 **0.379**，每張圖 2.6 ms。機車 0.766、電動三輪車 0.794、公車 0.776、汽車 0.765 都不錯；最差的是其他車輛（0.128，Recall 只有 0.08，這一類本來就是「不屬於任何類別的車」，定義很模糊）、紅綠燈（0.379）和動物（0.386）。完整數字見 [reports/idd-yolo26s-test.md](reports/idd-yolo26s-test.md)。

IDD 是印度的道路，招牌、車種（電動三輪車很多、汽車少）和台灣仍有差距，用在台灣畫面上的效果還沒有實際驗證過。

**起始權重的影響**：DOTA 是空拍影像，直覺上比 COCO 更接近 VisDrone，所以另外試了從 OBB 權重（`yolo26s-obb.pt`）開始微調：架構和其他設定完全相同，只換起始權重。backbone 和 neck 全部沿用，偵測頭只有類別輸出層重新學習（共轉移 696 / 708 個參數張量）。

```bash
python -m training.finetune --init yolo26s-obb.pt
```

| 起始權重 | 預訓練來源 | 最佳 epoch | test mAP50 | test mAP50-95 | val mAP50 |
| --- | --- | --- | --- | --- | --- |
| `yolo26s.pt` | COCO | 40 | 0.404 | 0.235 | 0.510 |
| `yolo26s-obb.pt` | Objects365 → DOTA | 41 | 0.408 | 0.237 | 0.503 |

- 差距都在 0.01 以內，而且方向不一致：test 上 OBB 起點略好，val 上 COCO 起點略好。只訓練一次無法排除隨機誤差，結論是**打平**；收斂速度也幾乎相同（最佳 epoch 40 vs 41）。
- 各類別互有高低：三輪車（0.257 → 0.291）、卡車（0.457 → 0.477）較好，腳踏車（0.190 → 0.177）、公車（0.619 → 0.605）較差。
- 推測原因：VisDrone 訓練集有 6,471 張、34 萬個標註，資料量夠大，微調後起始權重的差異幾乎被抹平；DOTA 多為衛星與航拍影像，拍攝高度比無人機高得多，視角仍有差距。
- 從 checkpoint 記錄的訓練參數看，Ultralytics 的 OBB 模型是先在 Objects365 上預訓練，再用 DOTA 訓練 50 個 epoch，所以這組比較同時換了預訓練資料集，不是單純的「COCO vs DOTA」。

App 的空拍模型維持 COCO 起點的版本。完整數字見 [reports/visdrone-yolo26s-obb-test.md](reports/visdrone-yolo26s-obb-test.md)、[reports/comparison-test.md](reports/comparison-test.md)。

訓練完成後，最佳權重會複製到 `weights/custom/`，App 的模型選單會自動出現對應選項，名稱由基礎模型和資料集組成，例如「YOLO26s-visdrone（空拍 / 俯視）」。也可以換成自己的資料集：`--data path/to/data.yaml`，格式請參考 [Ultralytics 資料集格式](https://docs.ultralytics.com/datasets/detect/)。

## 專案結構

```
app.py                  Streamlit 進入點與頁面導覽
views/                  各頁面 UI
traffic_vision/
  counting.py           越線計數（線段相交判斷，含方向）
  video.py              追蹤、軌跡繪製、H.264 輸出
  detection.py          單張影像偵測
  analytics.py          軌跡資料集的欄位正規化與統計
  models.py             模型清單與權重管理（含 weights/custom/ 的微調模型）
training/
  datasets.py           資料集下載與轉換
  finetune.py           微調
  evaluate.py           官方驗證器評估報告
  compare.py            多個模型的評估報告比較表
  crossdomain.py        COCO / VisDrone / IDD 跨資料集比較（預訓練 vs 微調）
  idd.py                IDD Detection 轉成 YOLO 格式（依影片序列切分）
  visdrone_eval.py      依 VisDrone 官方規則評估（忽略區域、官方計分）
reports/                評估報告
tests/                  單元測試（不需要 GPU 或模型權重）
```

UI 和運算邏輯是分開的：`traffic_vision/` 不依賴 Streamlit，可以直接在 notebook 或其他服務裡使用。

## 開發

```bash
pip install -r requirements-dev.txt
pytest -q
ruff check .
```

## 來源與授權

本專案是以 [Henryers/yolov8-traffic-app](https://github.com/Henryers/yolov8-traffic-app)（AGPL-3.0）為基礎重新設計、改寫的衍生作品，功能構想（偵測、軌跡、越線計數、資料集分析）沿用自原專案。

和原專案相比的主要差異：

- **介面**：PyQt5 桌面程式改成 Streamlit Web 介面，拿掉 MySQL 帳號登入，改成以本機上傳為主
- **架構**：原本 1,300 多行的單一檔案拆成彼此獨立、可以測試的模組；不再把整份 ultralytics 原始碼放進 repo，改成依賴官方套件，並升級到 YOLO26
- **越線計數**：原本只判斷「y 座標是否小於固定值」，物件只要出現在線的上方就會被計數；現在改成線段相交判斷，支援任意角度的計數線和 IN / OUT 方向，每個物件只計數一次
- **追蹤正確性**：每支影片會重置追蹤器，物件 ID 不會延續到下一支影片
- **影片輸出**：從 OpenCV mp4v 改成 H.264，不必再另外安裝 LAV Filters 解碼器
- **資料分析**：新增車流量時序圖、軌跡疊圖、移動距離 / 速度統計，同時相容原專案匯出的資料格式
- **工程**：單元測試、ruff、GitHub Actions CI、Docker；拿掉寫死在程式裡的資料庫密碼和本機路徑

依 AGPL-3.0 條款，本專案同樣以 [AGPL-3.0](LICENSE) 授權釋出。YOLO 模型與 [Ultralytics](https://github.com/ultralytics/ultralytics) 套件也是 AGPL-3.0 授權。

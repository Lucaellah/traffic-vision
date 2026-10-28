# 🚦 Traffic Vision

在本機執行的交通影像分析 Web 工具：上傳圖片或影片，就能用 **YOLO11** 做物件偵測、多目標追蹤、軌跡繪製和雙向越線計數，並把結果整理成可以下載、可以再分析的軌跡資料集。

> **所有資料都在你的電腦上處理。** 上傳的檔案只存在本機暫存資料夾，不會送到任何外部服務。

## 功能

| 頁面 | 功能 |
| --- | --- |
| 🖼️ 影像偵測 | 單張圖片的物件偵測或實例分割；可調整信心門檻、IoU、要偵測的類別；匯出偵測明細（CSV / Excel） |
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

**怎麼選模型**：路口監視器、行車記錄器這類平視畫面，用 YOLO11 一般模型（COCO 類別）；無人機空拍、俯視畫面，請選 **YOLO11-obb**（以 DOTA 空拍資料集訓練的旋轉框模型）。一般模型幾乎認不出從正上方拍的車。

### Docker（CPU）

```bash
docker build -t traffic-vision .
docker run --rm -p 8501:8501 traffic-vision
```

## 專案結構

```
app.py                  Streamlit 進入點與頁面導覽
views/                  各頁面 UI
traffic_vision/
  counting.py           越線計數（線段相交判斷，含方向）
  video.py              追蹤、軌跡繪製、H.264 輸出
  detection.py          單張影像偵測
  analytics.py          軌跡資料集的欄位正規化與統計
  models.py             模型清單與權重管理
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
- **架構**：原本 1,300 多行的單一檔案拆成彼此獨立、可以測試的模組；不再把整份 ultralytics 原始碼放進 repo，改成依賴官方套件，並升級到 YOLO11
- **越線計數**：原本只判斷「y 座標是否小於固定值」，物件只要出現在線的上方就會被計數；現在改成線段相交判斷，支援任意角度的計數線和 IN / OUT 方向，每個物件只計數一次
- **追蹤正確性**：每支影片會重置追蹤器，物件 ID 不會延續到下一支影片
- **影片輸出**：從 OpenCV mp4v 改成 H.264，不必再另外安裝 LAV Filters 解碼器
- **資料分析**：新增車流量時序圖、軌跡疊圖、移動距離 / 速度統計，同時相容原專案匯出的資料格式
- **工程**：單元測試、ruff、GitHub Actions CI、Docker；拿掉寫死在程式裡的資料庫密碼和本機路徑

依 AGPL-3.0 條款，本專案同樣以 [AGPL-3.0](LICENSE) 授權釋出。YOLO 模型與 [Ultralytics](https://github.com/ultralytics/ultralytics) 套件也是 AGPL-3.0 授權。

"""各頁面共用的 Streamlit 元件。"""

from __future__ import annotations

import hmac
import os
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from traffic_vision.labels import CATEGORIES, TRAFFIC_CATEGORIES, category_ids
from traffic_vision.models import (
    DEFAULT_MODEL,
    IMGSZ_OPTIONS,
    available_models,
    default_imgsz,
    load_model,
)


def password_gate() -> None:
    """有設定 TV_PASSWORD 環境變數時（部署到公開網址），要先輸入密碼才能使用。"""
    password = os.environ.get("TV_PASSWORD")
    if not password or st.session_state.get("authenticated"):
        return
    st.title(":material/traffic: Traffic Vision")
    with st.form("login"):
        entered = st.text_input("密碼", type="password")
        if st.form_submit_button("進入", icon=":material/login:"):
            if hmac.compare_digest(entered.encode(), password.encode()):
                st.session_state.authenticated = True
                st.rerun()
            time.sleep(1)  # 拖慢暴力猜密碼
            st.error("密碼錯誤")
    st.stop()


class LockedModel:
    """模型會被所有使用者共用，但 Ultralytics 的 predict 不是 thread-safe，一次只讓一個請求推論。"""

    def __init__(self, model):
        self._model = model
        self._lock = threading.Lock()

    def predict(self, *args, **kwargs):
        with self._lock:
            return self._model.predict(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._model, name)


@st.cache_resource(show_spinner="載入模型中（第一次使用會自動下載權重）…")
def cached_model(filename: str):
    return LockedModel(load_model(filename))


WORKDIR_PREFIX = "traffic-vision-"
WORKDIR_MAX_AGE_S = 24 * 3600


def workdir() -> Path:
    """每個瀏覽器 session 一個暫存資料夾，上傳檔與輸出都放這裡，不會離開執行網站的主機。"""
    if "workdir" not in st.session_state:
        remove_old_workdirs()
        st.session_state.workdir = Path(tempfile.mkdtemp(prefix=WORKDIR_PREFIX))
    return st.session_state.workdir


def remove_old_workdirs() -> None:
    """Streamlit 沒有 session 結束的通知，改成建立新資料夾時順便清掉一天以前的。"""
    cutoff = time.time() - WORKDIR_MAX_AGE_S
    for d in Path(tempfile.gettempdir()).glob(f"{WORKDIR_PREFIX}*"):
        try:
            if d.is_dir() and d.stat().st_mtime < cutoff:
                shutil.rmtree(d, ignore_errors=True)
        except OSError:
            pass


def save_upload(upload) -> Path:
    path = workdir() / upload.name
    path.write_bytes(upload.getbuffer())
    return path


# key 以 KEEP 開頭的元件，換頁後回來會保留原本的設定
KEEP = "keep_"


def keep_widget_state() -> None:
    """Streamlit 換頁時會清掉這次沒有顯示的元件狀態。

    每次執行前把這些值重新存一次，它們就會保留到重新整理網頁為止。
    """
    for k in list(st.session_state):
        if isinstance(k, str) and k.startswith(KEEP):
            st.session_state[k] = st.session_state[k]


@dataclass
class KeptFile:
    path: Path
    name: str
    file_id: str


def kept_upload(label: str, types: list[str], key: str) -> KeptFile | None:
    """上傳元件換頁後會變空，所以把檔案記在 session 裡，回來時沿用，直到上傳新檔或重新整理網頁。"""
    upload = st.file_uploader(label, type=types, key=f"{key}_uploader")
    store = f"{key}_file"
    kept: KeptFile | None = st.session_state.get(store)
    if upload is not None and (kept is None or kept.file_id != upload.file_id):
        kept = KeptFile(save_upload(upload), upload.name, upload.file_id)
        st.session_state[store] = kept
    if kept and upload is None:
        st.caption(f"目前使用：**{kept.name}**（上傳新檔案即可替換）")
    return kept


@dataclass
class ModelParams:
    model_file: str
    conf: float
    iou: float
    classes: list[int] | None
    imgsz: int | None  # None：沿用模型訓練時的解析度


def model_params(key: str, *, default_traffic: bool = True) -> ModelParams:
    """側邊欄的模型與參數設定。"""
    with st.sidebar:
        st.subheader("模型設定")
        models = available_models()
        names = list(models)
        label = st.selectbox("模型", names, index=names.index(DEFAULT_MODEL),
                             key=f"{KEEP}{key}_model",
                             help="無人機、高處俯拍的畫面請選「空拍 / 俯視」模型，"
                                  "其他畫面選一般模型")
        model_file = models[label]
        conf = st.slider("信心門檻 conf", 0.05, 0.95, 0.25, 0.05, key=f"{KEEP}{key}_conf",
                         help="越低偵測到的物件越多，但誤判也會增加")
        iou = st.slider("NMS IoU", 0.1, 0.95, 0.7, 0.05, key=f"{KEEP}{key}_iou",
                        help="重疊框的合併門檻")

        model = cached_model(model_file)
        auto = default_imgsz(model)
        imgsz = st.selectbox(
            "推論解析度", [None, *IMGSZ_OPTIONS], key=f"{KEEP}{key}_imgsz",
            format_func=lambda s: f"自動（{auto}）" if s is None else str(s),
            help="一般用「自動」即可；空拍畫面建議 1536。越大越能抓到小物體，但速度越慢")

        picked = st.multiselect("只偵測這些類別（留空 = 全部）", list(CATEGORIES),
                                default=TRAFFIC_CATEGORIES if default_traffic else [],
                                key=f"{KEEP}{key}_classes")
        classes = category_ids(model.names, picked) if picked else None
        missing = [c for c in picked if not category_ids(model.names, [c])]
        if missing:
            st.caption(f"這個模型沒有：{'、'.join(missing)}")
    return ModelParams(model_file, conf, iou, classes, imgsz)


def download_table(df, stem: str, key: str) -> None:
    import io

    c1, c2 = st.columns(2)
    c1.download_button("下載 CSV", df.to_csv(index=False).encode("utf-8-sig"),
                       f"{stem}.csv", "text/csv", key=f"{key}_csv", width="stretch")
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    c2.download_button("下載 Excel", buf.getvalue(), f"{stem}.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       key=f"{key}_xlsx", width="stretch")

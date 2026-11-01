"""各頁面共用的 Streamlit 元件。"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from traffic_vision.labels import TRAFFIC_CLASSES, display_name
from traffic_vision.models import available_models, load_model


@st.cache_resource(show_spinner="載入模型中（第一次使用會自動下載權重）…")
def cached_model(filename: str):
    return load_model(filename)


def workdir() -> Path:
    """每個瀏覽器 session 一個暫存資料夾，上傳檔與輸出都放這裡，不會離開本機。"""
    if "workdir" not in st.session_state:
        st.session_state.workdir = Path(tempfile.mkdtemp(prefix="traffic-vision-"))
    return st.session_state.workdir


def save_upload(upload) -> Path:
    path = workdir() / upload.name
    path.write_bytes(upload.getbuffer())
    return path


@dataclass
class ModelParams:
    model_file: str
    conf: float
    iou: float
    classes: list[int] | None


def model_params(key: str, *, default_traffic: bool = True) -> ModelParams:
    """側邊欄的模型與參數設定。"""
    with st.sidebar:
        st.subheader("模型設定")
        models = available_models()
        label = st.selectbox("模型", list(models), key=f"{key}_model")
        model_file = models[label]
        conf = st.slider("信心門檻 conf", 0.05, 0.95, 0.25, 0.05, key=f"{key}_conf",
                         help="越低偵測到的物件越多，但誤判也會增加")
        iou = st.slider("NMS IoU", 0.1, 0.95, 0.7, 0.05, key=f"{key}_iou",
                        help="重疊框的合併門檻")

        names = cached_model(model_file).names
        name_to_id = {n: i for i, n in names.items()}
        default = [n for n in TRAFFIC_CLASSES if n in name_to_id] if default_traffic else []
        picked = st.multiselect("只偵測這些類別（留空 = 全部）", list(name_to_id),
                                default=default, format_func=display_name,
                                key=f"{key}_classes_{model_file}")  # 換模型時類別清單也不同
    classes = [name_to_id[n] for n in picked] or None
    return ModelParams(model_file, conf, iou, classes)


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

import altair as alt
import cv2
import numpy as np
import streamlit as st

from traffic_vision.detection import detect_image
from views.common import cached_model, download_table, model_params

st.title(":material/image_search: 影像偵測")
params = model_params("image", default_traffic=False)

upload = st.file_uploader("上傳圖片", type=["jpg", "jpeg", "png", "bmp", "webp"])
if upload is None:
    st.caption("支援 JPG、PNG、BMP、WebP。")
    st.stop()

image = cv2.imdecode(np.frombuffer(upload.getvalue(), np.uint8), cv2.IMREAD_COLOR)
if image is None:
    st.error("無法讀取這張圖片，請確認檔案格式。")
    st.stop()

model = cached_model(params.model_file)
with st.spinner("偵測中…"):
    annotated, detections = detect_image(model, image, conf=params.conf, iou=params.iou,
                                         classes=params.classes, imgsz=params.imgsz)

left, right = st.columns(2)
left.image(image[:, :, ::-1], caption="原始圖片", width="stretch")
right.image(annotated, caption=f"偵測結果：{len(detections)} 個物件", width="stretch")

if detections.empty:
    st.warning("沒有偵測到物件，可以試著降低信心門檻或放寬類別。")
    st.stop()

st.subheader("類別統計")
counts = detections["class_name"].value_counts().rename_axis("類別").reset_index(name="物件數")
st.altair_chart(
    alt.Chart(counts).mark_bar().encode(
        x=alt.X("類別:N", sort=None, axis=alt.Axis(labelAngle=0)),  # 類別名稱橫著寫
        y="物件數:Q",
    ),
    width="stretch",
)

st.subheader("偵測明細")
st.dataframe(detections.round(3), width="stretch", hide_index=True)
download_table(detections, upload.name.rsplit(".", 1)[0] + "_detections", "image")

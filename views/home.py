import streamlit as st
import torch

st.title(":material/traffic: Traffic Vision")
st.markdown(
    "以 **YOLO26** 為核心的交通影像分析工具。"
    "上傳圖片或影片進行物件偵測、多目標追蹤與越線計數，(要改：再把產生的軌跡資料集做統計與視覺化。）"
)

# 隱私說明縮成 70%：zoom 會連同文字、圖示和內距一起縮小
st.html("<style>.st-key-privacy [data-testid='stAlert'] { zoom: 0.7; }</style>")
with st.container(key="privacy"):
    st.info(
        "**所有資料都在你的電腦上處理。** 上傳的檔案只存在本機暫存資料夾，"
        "不會傳送到任何外部伺服器（第一次使用某個模型時，只會下載公開的模型權重）。",
        icon=":material/lock:",
    )

FEATURES = [
    ("views/image.py", ":material/image_search:", "影像偵測",
     "單張圖片的物件偵測或實例分割，可匯出偵測結果。"),
    ("views/video.py", ":material/videocam:", "影片追蹤與計數",
     "多目標追蹤、軌跡繪製、自訂計數線的雙向越線計數。"),
    ("views/analysis.py", ":material/bar_chart:", "資料分析",
     "類別統計、位置熱力圖、單一物件軌跡與速度分析。"),
]

for col, (page, icon, title, desc) in zip(st.columns(3), FEATURES, strict=True):
    with col, st.container(border=True, height="stretch"):
        st.subheader(f"{icon} {title}")
        st.write(desc)
        if st.button("前往", icon=":material/arrow_forward:", key=page, width="stretch"):
            st.switch_page(page)

st.divider()
device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU（未偵測到可用 GPU）"
st.caption(f"運算裝置：{device}")

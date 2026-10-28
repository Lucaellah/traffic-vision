import streamlit as st
import torch

st.title("🚦 Traffic Vision")
st.markdown(
    "以 **YOLO11** 為核心的交通影像分析工具。上傳圖片或影片進行物件偵測、多目標追蹤與越線計數，"
    "再把產生的軌跡資料集做統計與視覺化。"
)

st.info(
    "**所有資料都在你的電腦上處理。** 上傳的檔案只存在本機暫存資料夾，"
    "不會傳送到任何外部伺服器（第一次使用某個模型時，只會下載公開的模型權重）。",
    icon="🔒",
)

c1, c2, c3 = st.columns(3)
with c1:
    st.subheader("🖼️ 影像偵測")
    st.write("單張圖片的物件偵測或實例分割，可匯出偵測結果。")
    st.page_link("views/image.py", label="前往", icon="➡️")
with c2:
    st.subheader("🎞️ 影片追蹤與計數")
    st.write("多目標追蹤、軌跡繪製、自訂計數線的雙向越線計數。")
    st.page_link("views/video.py", label="前往", icon="➡️")
with c3:
    st.subheader("📊 資料分析")
    st.write("類別統計、位置熱力圖、單一物件軌跡與速度分析。")
    st.page_link("views/analysis.py", label="前往", icon="➡️")

st.divider()
device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU（未偵測到可用 GPU）"
st.caption(f"運算裝置：{device}")

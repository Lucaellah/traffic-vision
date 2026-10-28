"""Streamlit 進入點：streamlit run app.py"""

import streamlit as st

st.set_page_config(page_title="Traffic Vision", page_icon="🚦", layout="wide")

pages = [
    st.Page("views/home.py", title="首頁", icon="🏠", default=True),
    st.Page("views/image.py", title="影像偵測", icon="🖼️"),
    st.Page("views/video.py", title="影片追蹤與計數", icon="🎞️"),
    st.Page("views/analysis.py", title="資料分析", icon="📊"),
]
st.navigation(pages).run()

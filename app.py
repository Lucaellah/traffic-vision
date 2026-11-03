"""Streamlit 進入點：streamlit run app.py"""

import streamlit as st

from views.common import keep_widget_state

st.set_page_config(page_title="Traffic Vision", page_icon=":material/traffic:", layout="wide")
keep_widget_state()

pages = [
    st.Page("views/home.py", title="首頁", icon=":material/home:", default=True),
    st.Page("views/image.py", title="影像偵測", icon=":material/image_search:"),
    st.Page("views/video.py", title="影片追蹤與計數", icon=":material/videocam:"),
    st.Page("views/analysis.py", title="資料分析", icon=":material/bar_chart:"),
]
st.navigation(pages).run()

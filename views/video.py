import streamlit as st

from traffic_vision import analytics
from traffic_vision.counting import LineCounter
from traffic_vision.labels import display_name
from traffic_vision.models import load_model
from traffic_vision.video import draw_line, probe, process_video, read_first_frame
from views.common import download_table, model_params, save_upload, workdir

st.title("🎞️ 影片追蹤與計數")
params = model_params("video")

upload = st.file_uploader("上傳影片", type=["mp4", "mov", "avi", "mkv", "webm"])
if upload is None:
    st.caption("支援 MP4、MOV、AVI、MKV、WebM。處理後的影片、軌跡資料都可以下載。")
    st.stop()

src = save_upload(upload)
try:
    info = probe(src)
    first = read_first_frame(src)
except ValueError as e:
    st.error(str(e))
    st.stop()

st.caption(f"{info.width}×{info.height}・{info.fps:.1f} fps・{info.frames} 幀"
           f"（約 {info.duration_s:.1f} 秒）")

with st.sidebar:
    st.subheader("處理範圍")
    max_s = max(1, int(info.duration_s) + 1)
    seconds = st.slider("最多處理幾秒", 1, max_s, min(max_s, 30),
                        help="長影片可以先處理一小段確認參數")
    trail = st.slider("軌跡長度（幀）", 10, 300, 60, 10)
    show_labels = st.toggle("影片中顯示類別與 ID 標籤", value=True,
                            help="物件很多時（例如空拍）關掉會比較清楚")

st.subheader("計數線")
use_line = st.toggle("啟用越線計數", value=True)
line = None
if use_line:
    st.caption("用畫面寬高的百分比設定線段兩端點。線上的箭頭指向 IN 的方向，"
               "反方向穿越算 OUT；對調起點和終點即可反轉方向。")
    c1, c2 = st.columns(2)
    x1 = c1.slider("起點 X %", 0, 100, 0)
    y1 = c1.slider("起點 Y %", 0, 100, 60)
    x2 = c2.slider("終點 X %", 0, 100, 100)
    y2 = c2.slider("終點 Y %", 0, 100, 60)
    line = (
        (info.width * x1 / 100, info.height * y1 / 100),
        (info.width * x2 / 100, info.height * y2 / 100),
    )
    preview = first.copy()
    draw_line(preview, LineCounter(*line))
    st.image(preview[:, :, ::-1], caption="第一幀預覽", width="stretch")

if st.button("開始分析", type="primary", width="stretch"):
    bar = st.progress(0.0, text="準備中…")

    def on_progress(done: int, total: int) -> None:
        frac = done / total if total else 0.0
        bar.progress(min(frac, 1.0), text=f"處理中 {done}/{total} 幀")

    dst = workdir() / f"{src.stem}_tracked.mp4"
    try:
        result = process_video(
            src, dst, load_model(params.model_file),
            conf=params.conf, iou=params.iou, classes=params.classes, line=line,
            max_frames=int(seconds * info.fps), trail_length=trail,
            show_labels=show_labels, progress=on_progress,
        )
    except Exception as e:  # 顯示給使用者，而不是讓整頁當掉
        bar.empty()
        st.exception(e)
        st.stop()
    bar.empty()
    st.session_state.video_result = result
    st.session_state.video_name = src.stem
    # 給「資料分析」頁直接使用
    st.session_state.tracks = result.tracks
    st.session_state.tracks_source = f"{upload.name}（影片追蹤結果）"

result = st.session_state.get("video_result")
if result is None or st.session_state.get("video_name") != src.stem:
    st.stop()

st.divider()
st.subheader("分析結果")
tracks = result.tracks
m = st.columns(4)
m[0].metric("處理幀數", result.processed_frames)
m[1].metric("追蹤到的物件", tracks["track_id"].nunique() if not tracks.empty else 0)
if result.counter:
    m[2].metric("IN", result.counter.count_in)
    m[3].metric("OUT", result.counter.count_out)

st.video(str(result.output_path))
st.download_button("下載標註影片", result.output_path.read_bytes(),
                   result.output_path.name, "video/mp4", width="stretch")

if tracks.empty:
    st.warning("沒有追蹤到任何物件。")
    st.stop()

counts = analytics.class_counts(tracks).set_index("class_name")["count"]
counts.index = counts.index.map(display_name)
st.bar_chart(counts, horizontal=True)

st.subheader("軌跡資料集")
st.caption("每一列是某個物件在某一幀的位置（框中心點，像素座標）。")
st.dataframe(tracks.round(3).head(500), width="stretch", hide_index=True)
download_table(tracks, f"{st.session_state.video_name}_tracks", "video")
st.page_link("views/analysis.py", label="用這份資料進行分析", icon="📊")

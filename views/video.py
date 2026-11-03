import streamlit as st

from traffic_vision import analytics
from traffic_vision.counting import LineCounter
from traffic_vision.labels import display_name
from traffic_vision.video import (
    draw_line,
    draw_speed_line,
    probe,
    process_video,
    read_first_frame,
)
from views.common import KEEP, cached_model, download_table, kept_upload, model_params, workdir

st.title(":material/videocam: 影片追蹤與計數")
params = model_params("video")

upload = kept_upload("上傳影片", ["mp4", "mov", "avi", "mkv", "webm"], "video")
if upload is None:
    st.caption("支援 MP4、MOV、AVI、MKV、WebM。處理後的影片、軌跡資料都可以下載。")
    st.stop()

src = upload.path
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
    seconds = st.slider("最多處理幾秒", 1, max_s, min(max_s, 30), key=f"{KEEP}video_seconds",
                        help="長影片可以先處理一小段確認參數")
    trail = st.slider("軌跡長度（幀）", 10, 300, 60, 10, key=f"{KEEP}video_trail")
    show_labels = st.toggle("影片中顯示類別與 ID 標籤", value=True, key=f"{KEEP}video_labels",
                            help="物件很多時（例如空拍）關掉會比較清楚")

MODES = {
    "fixed": "固定攝影機（路口 CCTV、懸停空拍）",
    "dashcam": "行車紀錄器（攝影機跟著車移動）",
}
mode = st.radio("拍攝方式", list(MODES), format_func=MODES.get, horizontal=True,
                key=f"{KEEP}video_mode",
                help="會影響可以做的分析：固定攝影機看路段的車流量與車速，"
                     "行車紀錄器看沿途的擁擠程度與前方接近事件")


def line_sliders(key: str, y_default: int) -> tuple:
    c1, c2 = st.columns(2)
    x1 = c1.slider("起點 X %", 0, 100, 0, key=f"{KEEP}{key}_x1")
    y1 = c1.slider("起點 Y %", 0, 100, y_default, key=f"{KEEP}{key}_y1")
    x2 = c2.slider("終點 X %", 0, 100, 100, key=f"{KEEP}{key}_x2")
    y2 = c2.slider("終點 Y %", 0, 100, y_default, key=f"{KEEP}{key}_y2")
    return (
        (info.width * x1 / 100, info.height * y1 / 100),
        (info.width * x2 / 100, info.height * y2 / 100),
    )


line = speed_line = None
speed_distance = None
if mode == "fixed":
    st.subheader("計數線")
    if st.toggle("啟用越線計數", value=True, key=f"{KEEP}video_use_line"):
        st.caption("用畫面寬高的百分比設定線段兩端點。線上的箭頭指向 IN 的方向，"
                   "反方向穿越算 OUT；對調起點和終點即可反轉方向。")
        line = line_sliders("line", 60)

        if st.toggle("同時測速", value=False, key=f"{KEEP}video_use_speed",
                     help="在同一條路上再畫一條測速線，量出兩條線之間的實際距離。"
                          "車子通過兩條線的時間差就能換算成 km/h"):
            st.caption("測速線（橘色）要和計數線一樣橫跨車道。國道車道虛線一段 4 公尺、"
                       "間隔 6 公尺，一組 10 公尺，可以拿來估算距離；"
                       "低幀率影片（如 CCTV）兩線距離建議 30 公尺以上，誤差比較小。")
            speed_line = line_sliders("speed", 45)
            speed_distance = st.number_input("兩條線之間的實際距離（公尺）", 1.0, 500.0, 30.0, 5.0,
                                             key=f"{KEEP}video_speed_distance")

        preview = first.copy()
        if speed_line:
            draw_speed_line(preview, LineCounter(*speed_line))
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
            src, dst, cached_model(params.model_file),
            conf=params.conf, iou=params.iou, classes=params.classes, imgsz=params.imgsz,
            line=line, speed_line=speed_line,
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
    st.session_state.video_analysis = {
        "mode": mode,
        "crossings": result.crossings,
        "speed_distance_m": speed_distance,
        "width": info.width,
        "height": info.height,
        "frames": result.processed_frames,
        "duration_s": result.processed_frames / info.fps,
        "video_path": str(result.output_path),
    }

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
st.page_link("views/analysis.py", label="用這份資料進行分析", icon=":material/bar_chart:")

import cv2
import pandas as pd
import plotly.express as px
import streamlit as st

from traffic_vision import analytics
from traffic_vision.labels import VRU_CATEGORIES, display_name
from views.common import KEEP, download_table, kept_upload

st.title(":material/bar_chart: 資料分析")

MODES = {"fixed": "固定攝影機", "dashcam": "行車紀錄器"}
DIRECTION_ZH = {"in": "IN", "out": "OUT"}

sources = ["上傳 CSV / Excel"]
if "tracks" in st.session_state:
    sources.insert(0, st.session_state.tracks_source)
source = st.radio("資料來源", sources, horizontal=True, key=f"{KEEP}analysis_source")

video = None
if source == "上傳 CSV / Excel":
    upload = kept_upload("上傳軌跡資料集", ["csv", "xlsx", "xls"], "analysis")
    st.caption("需要 track_id、x、y 欄位；frame、time_s、class_name、w、h 可選。"
               "也支援原專案匯出的 Track ID / X / Y / Category 格式。")
    if upload is None:
        st.stop()
    try:
        raw = analytics.read_table(upload.path, upload.name)
    except Exception as e:
        st.error(f"讀取失敗：{e}")
        st.stop()
    mode = st.radio("拍攝方式", list(MODES), format_func=MODES.get, horizontal=True,
                    key=f"{KEEP}analysis_mode")
else:
    raw = st.session_state.tracks
    video = st.session_state.get("video_analysis")
    mode = video["mode"] if video else "fixed"
    st.caption(f"拍攝方式：{MODES[mode]}（在影片頁設定）")

try:
    df = analytics.normalize(raw)
except ValueError as e:
    st.error(str(e))
    st.stop()
if df.empty:
    st.warning("資料是空的。")
    st.stop()

df["label"] = df["class_name"].map(display_name)
ov = analytics.overview(df)
duration = video["duration_s"] if video else ov.get("duration_s", 0.0)
has_time = "time_s" in df.columns and df["time_s"].notna().any()


def raw_data_tab() -> None:
    """原始軌跡資料：單一物件與物件明細。"""
    st.caption(f"共 {ov['rows']:,} 筆位置紀錄、{ov['tracks']} 個追蹤 ID。"
               "同一個物件暫時沒被偵測到時可能會換一個新 ID，所以 ID 數會比實際數量多。")
    summary = analytics.track_summary(df)
    summary["label"] = summary["class_name"].map(display_name)
    by_id = summary.set_index("track_id")
    tid = st.selectbox("物件 ID（依出現時間長短排序）", by_id.index.tolist(),
                       format_func=lambda i: f"#{i}・{by_id.at[i, 'label']}")
    one = df[df["track_id"] == tid]
    row = by_id.loc[tid]
    c = st.columns(3)
    c[0].metric("出現次數", int(row["points"]))
    c[1].metric("移動距離", f"{row['distance_px']:.0f} px")
    if "speed_px_s" in row and row["speed_px_s"] == row["speed_px_s"]:
        c[2].metric("平均速度", f"{row['speed_px_s']:.0f} px/s")
    fig = px.scatter(one, x="x", y="y", color="frame", color_continuous_scale="Viridis")
    fig.add_scatter(x=one["x"], y=one["y"], mode="lines", line=dict(color="gray", width=1),
                    showlegend=False)
    fig.update_xaxes(range=[0, df["x"].max() * 1.02])
    fig.update_yaxes(range=[df["y"].max() * 1.02, 0], scaleanchor="x")
    st.plotly_chart(fig, width="stretch")

    st.caption("距離與速度都是像素單位。")
    st.dataframe(summary.drop(columns="class_name"), width="stretch", hide_index=True)
    download_table(summary, "track_summary", "summary")


def heatmap_tab() -> None:
    st.caption("車輛中心點在畫面上的分布，可以看出哪些車道比較常被使用。")
    bins = st.slider("網格密度", 10, 120, 50, 5, key=f"{KEEP}analysis_bins")
    fig = px.density_heatmap(df, x="x", y="y", nbinsx=bins, nbinsy=bins,
                             color_continuous_scale="Inferno")
    fig.update_yaxes(autorange="reversed", scaleanchor="x")  # 影像座標 y 軸向下
    st.plotly_chart(fig, width="stretch")


# ---------------- 固定攝影機 ----------------

def fixed_view() -> None:
    crossings = video["crossings"] if video else None
    counted = (analytics.vehicle_crossings(crossings, df)
               if crossings is not None and not crossings.empty else None)
    if counted is None or counted.empty:
        st.info("沒有越線紀錄。車流量、車種組成和車速都以「越過計數線的車輛」計算，"
                "請在影片頁開啟越線計數後重新分析。")
        tab_heat, tab_raw = st.tabs(["位置熱力圖", "原始資料"])
        with tab_heat:
            heatmap_tab()
        with tab_raw:
            raw_data_tab()
        return

    speeds = (analytics.crossing_speeds(crossings, video["speed_distance_m"])
              if video.get("speed_distance_m") else None)
    n_in = int((counted["direction"] == "in").sum())
    n_out = int((counted["direction"] == "out").sum())

    m = st.columns(4)
    m[0].metric("越線車輛", len(counted), help="每台車只算一次")
    m[0].caption(f"IN {n_in}・OUT {n_out}")
    m[1].metric("換算流量", f"{analytics.hourly_rate(len(counted), duration):,.0f} 輛/時",
                help=f"以這段 {duration:.0f} 秒影片的車輛數換算成每小時；影片越長越準")
    share = analytics.heavy_share(counted["category"])
    m[2].metric("大型車比例", f"{share:.0%}", help="公車與卡車占越線車輛的比例")
    if speeds is not None and not speeds.empty:
        m[3].metric("平均車速", f"{speeds['speed_kmh'].mean():.0f} km/h",
                    help=f"{len(speeds)} 台車同時通過計數線與測速線")
    else:
        m[3].metric("平均車速", "—", help="在影片頁開啟「同時測速」才會計算")

    tab_flow, tab_class, tab_speed, tab_heat, tab_raw = st.tabs(
        ["車流量", "車種組成", "車速", "位置熱力圖", "原始資料"])

    with tab_flow:
        bin_s = 10 if duration <= 180 else 60
        flow = analytics.flow_series(counted, bin_s)
        flow["方向"] = flow["direction"].map(DIRECTION_ZH)
        unit = "秒" if bin_s < 60 else "分鐘"
        flow["時間"] = flow["start_s"] / (1 if bin_s < 60 else 60)
        fig = px.bar(flow, x="時間", y="count", color="方向", barmode="group",
                     labels={"count": "越線車輛數", "時間": f"時間（{unit}）"},
                     title=f"每 {bin_s if bin_s < 60 else 1} {unit}的越線車輛數")
        st.plotly_chart(fig, width="stretch")

    with tab_class:
        comp = counted.groupby(["category", "direction"]).size().rename("count").reset_index()
        comp["方向"] = comp["direction"].map(DIRECTION_ZH)
        c1, c2 = st.columns([3, 2])
        c1.plotly_chart(
            px.bar(comp, x="category", y="count", color="方向", text_auto=True,
                   labels={"category": "類別", "count": "越線車輛數"}).update_xaxes(tickangle=0),
            width="stretch")
        total = comp.groupby("category")["count"].sum().reset_index()
        c2.plotly_chart(px.pie(total, names="category", values="count", hole=0.4),
                        width="stretch")

    with tab_speed:
        if speeds is None or speeds.empty:
            st.info("在影片頁開啟「同時測速」，畫一條測速線並輸入兩線之間的實際距離，"
                    "就能算出每台車的車速。")
        else:
            speeds = speeds.merge(counted[["track_id", "direction", "category"]], on="track_id")
            speeds["方向"] = speeds["direction"].map(DIRECTION_ZH)
            s = st.columns(3)
            s[0].metric("測到車速的車輛", len(speeds))
            s[1].metric("中位數", f"{speeds['speed_kmh'].median():.0f} km/h")
            s[2].metric("85 百分位", f"{speeds['speed_kmh'].quantile(0.85):.0f} km/h",
                        help="85% 的車速度在這以下，交通工程常用來訂定速限")
            st.plotly_chart(
                px.histogram(speeds, x="speed_kmh", color="方向", nbins=20, barmode="overlay",
                             labels={"speed_kmh": "車速（km/h）"}),
                width="stretch")
            st.caption(f"兩線距離 {video['speed_distance_m']:.0f} 公尺。"
                       "低幀率影片的時間誤差較大，個別車速僅供參考，平均值比較可靠。")
            st.dataframe(speeds[["track_id", "category", "方向", "dt_s", "speed_kmh"]].round(2),
                         width="stretch", hide_index=True)

    with tab_heat:
        heatmap_tab()
    with tab_raw:
        raw_data_tab()


# ---------------- 行車紀錄器 ----------------

def event_snapshots(events, path: str) -> None:
    """從標註影片擷取事件當下的畫面，並用紅框標出接近的物件。"""
    cap = cv2.VideoCapture(path)
    try:
        cols = st.columns(3)
        for i, ev in enumerate(events.head(9).itertuples()):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(ev.frame))
            ok, frame = cap.read()
            if not ok:
                continue
            p1 = (int(ev.x - ev.w / 2), int(ev.y - ev.h / 2))
            p2 = (int(ev.x + ev.w / 2), int(ev.y + ev.h / 2))
            cv2.rectangle(frame, p1, p2, (0, 0, 255), max(3, frame.shape[1] // 300))
            cols[i % 3].image(frame[:, :, ::-1], width="stretch",
                              caption=f"{ev.time_s:.1f} 秒・{ev.category}・TTC {ev.ttc_s:.1f} 秒")
    finally:
        cap.release()


def dashcam_view() -> None:
    if not has_time or not {"w", "h"} <= set(df.columns):
        st.warning("行車紀錄器分析需要 time_s、w、h 欄位（影片頁產生的資料都有）。")
        raw_data_tab()
        return
    width = video["width"] if video else float((df["x"] + df["w"] / 2).max())
    frames = video["frames"] if video else int(df["frame"].max()) + 1

    presence = analytics.presence_series(df, 1.0)
    vehicles = presence[~presence["category"].isin(VRU_CATEGORIES)]
    per_second = vehicles.groupby("start_s")["count"].sum()
    vru = analytics.exposure(df, frames, VRU_CATEGORIES)

    with st.sidebar:
        st.subheader("接近事件")
        threshold = st.slider("TTC 門檻（秒）", 0.5, 3.0, 1.5, 0.1, key=f"{KEEP}analysis_ttc",
                              help="碰撞時間低於這個值才算接近事件。一般前方碰撞警示約 1.5～2.7 秒")
        center = st.slider("前方範圍（畫面中央寬度 %）", 20, 100, 40, 5,
                           key=f"{KEEP}analysis_center",
                           help="只看畫面中央這個寬度內的物件，排除旁邊車道和對向來車") / 100
    events = analytics.ttc_events(df, width, threshold_s=threshold, center_frac=center)

    m = st.columns(4)
    m[0].metric("平均畫面車輛數", f"{per_second.mean():.1f}", help="同一時間畫面中平均有幾台車")
    m[1].metric("最擁擠時刻", f"{per_second.idxmax():.0f} 秒",
                help=f"畫面中平均有 {per_second.max():.1f} 台車")
    m[2].metric("行人出現時間", f"{vru['人']:.0%}", help="畫面中有行人的時間占比")
    m[3].metric("接近事件", len(events), help=f"前方物件碰撞時間低於 {threshold:.1f} 秒")

    tab_crowd, tab_vru, tab_ttc, tab_raw = st.tabs(
        ["擁擠程度", "弱勢用路人", "接近事件", "原始資料"])

    with tab_crowd:
        st.caption("每秒平均畫面中有幾個物件，可以比較沿途哪一段路最擁擠。")
                   
        st.plotly_chart(
            px.area(presence, x="start_s", y="count", color="category",
                    labels={"start_s": "時間（秒）", "count": "畫面中的數量", "category": "類別"}),
            width="stretch")

    with tab_vru:
        st.caption("行人、腳踏車、機車在畫面中出現的時間占比。這些用路人沒有車體保護，"
                   "出現越頻繁，駕駛越需要注意。")
        share = pd.DataFrame({"類別": vru.index, "占比": vru.to_numpy()})
        st.plotly_chart(
            px.bar(share, x="類別", y="占比", text_auto=".0%")
            .update_yaxes(tickformat=".0%", range=[0, 1]).update_xaxes(tickangle=0),
            width="stretch")
        st.plotly_chart(
            px.line(presence[presence["category"].isin(VRU_CATEGORIES)],
                    x="start_s", y="count", color="category",
                    labels={"start_s": "時間（秒）", "count": "畫面中的數量", "category": "類別"}),
            width="stretch")

    with tab_ttc:
        st.caption("TTC（碰撞時間）由前方物件的框放大速度估計：框越快變大，代表接近得越快。"
                   "不需要知道真實距離，但假設相對速度固定，自己的車正在煞車時會高估風險，"
                   "適合用來快速挑出需要回放檢查的片段。")
        if events.empty:
            st.success(f"沒有 TTC 低於 {threshold:.1f} 秒的接近事件。")
        else:
            if video and video.get("video_path"):
                event_snapshots(events, video["video_path"])
            table = events.assign(類別=events["category"])[
                ["time_s", "類別", "ttc_s", "track_id"]].round(2)
            st.dataframe(table.rename(columns={"time_s": "時間（秒）", "ttc_s": "TTC（秒）"}),
                         width="stretch", hide_index=True)
            download_table(events, "ttc_events", "ttc")

    with tab_raw:
        raw_data_tab()


if mode == "fixed":
    fixed_view()
else:
    dashcam_view()

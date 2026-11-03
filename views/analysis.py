import plotly.express as px
import streamlit as st

from traffic_vision import analytics
from traffic_vision.labels import display_name
from views.common import download_table

st.title(":material/bar_chart: 資料分析")

sources = ["上傳 CSV / Excel"]
if "tracks" in st.session_state:
    sources.insert(0, st.session_state.tracks_source)
source = st.radio("資料來源", sources, horizontal=True)

if source == "上傳 CSV / Excel":
    upload = st.file_uploader("上傳軌跡資料集", type=["csv", "xlsx", "xls"])
    st.caption("需要 track_id、x、y 欄位；frame、time_s、class_name 可選。"
               "也支援原專案匯出的 Track ID / X / Y / Category 格式。")
    if upload is None:
        st.stop()
    try:
        raw = analytics.read_table(upload, upload.name)
    except Exception as e:
        st.error(f"讀取失敗：{e}")
        st.stop()
else:
    raw = st.session_state.tracks

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
m = st.columns(4)
m[0].metric("資料筆數", f"{ov['rows']:,}")
m[1].metric("物件數", ov["tracks"])
m[2].metric("類別數", ov["classes"])
span = f"{ov['duration_s']:.1f} 秒" if "duration_s" in ov else f"{ov['frames']} 幀"
m[3].metric("時間長度", span)

tab_overview, tab_heat, tab_track, tab_table = st.tabs(
    ["類別統計", "位置熱力圖", "單一物件", "物件明細"]
)

with tab_overview:
    counts = analytics.class_counts(df)
    counts["label"] = counts["class_name"].map(display_name)
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(
        px.bar(counts, x="label", y="count", labels={"label": "類別", "count": "物件數"},
               text_auto=True).update_xaxes(tickangle=0),
        width="stretch",
    )
    c2.plotly_chart(px.pie(counts, names="label", values="count", hole=0.4), width="stretch")

    if "time_s" in df.columns:
        first_seen = df.groupby("track_id").agg(time_s=("time_s", "min"), label=("label", "first"))
        first_seen["秒"] = first_seen["time_s"].astype(int)
        flow = first_seen.groupby(["秒", "label"]).size().rename("新出現物件").reset_index()
        st.plotly_chart(
            px.bar(flow, x="秒", y="新出現物件", color="label", title="每秒新出現的物件（車流量）"),
            width="stretch",
        )

with tab_heat:
    bins = st.slider("網格密度", 10, 120, 50, 5)
    fig = px.density_heatmap(df, x="x", y="y", nbinsx=bins, nbinsy=bins,
                             color_continuous_scale="Inferno")
    fig.update_yaxes(autorange="reversed", scaleanchor="x")  # 影像座標 y 軸向下
    st.plotly_chart(fig, width="stretch")

    top = df[df["track_id"].isin(df["track_id"].value_counts().head(150).index)]
    fig = px.line(top, x="x", y="y", color="label", line_group="track_id",
                  title="主要物件軌跡（出現最久的 150 個）")
    fig.update_traces(opacity=0.5)
    fig.update_yaxes(autorange="reversed", scaleanchor="x")
    st.plotly_chart(fig, width="stretch")

summary = analytics.track_summary(df)
summary["label"] = summary["class_name"].map(display_name)

with tab_track:
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

with tab_table:
    st.caption("距離與速度都是像素單位；若要換算成公尺，需要依攝影機架設位置做校正。")
    st.dataframe(summary.drop(columns="class_name"), width="stretch", hide_index=True)
    download_table(summary, "track_summary", "summary")

from __future__ import annotations

from datetime import datetime
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from google_source import download_google_sheet
from kpi_data import load_kpi_data, period_detail


APP_DIR = Path(__file__).resolve().parent
SAMPLE_FILE = APP_DIR / "data" / "DCKPI Dashboard.xlsx"
GOOGLE_SHEET_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1msT7WcjkYva2yojVNrET2xJfqRS00xH85j1D85Ne36Q/edit?usp=sharing"
)
GREEN = "#54B435"
DARK_GREEN = "#1F7A3D"
YELLOW = "#F2C94C"
RED = "#EB5757"
BLUE = "#2F80ED"
INK = "#16302B"
MUTED = "#6B7C78"
DATA_SCHEMA_VERSION = 4


st.set_page_config(
    page_title="ZX R&D KPI Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp { background: #F5F8F6; color: #16302B; }
      [data-testid="stSidebar"] { background: #12372A; }
      [data-testid="stSidebar"] * { color: #F4FFF7; }
      [data-testid="stSidebar"] .stButton button,
      [data-testid="stSidebar"] .stDownloadButton button {
        background: #E8F5EC; color: #12372A; border: 0;
      }
      [data-testid="stSidebar"] .stButton button *,
      [data-testid="stSidebar"] .stDownloadButton button * {
        color: #12372A !important;
      }
      [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        background: #F4F8F5; border-color: #A9C4B4;
      }
      [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * {
        color: #315548 !important;
      }
      [data-testid="stMetric"] {
        background: white; border: 1px solid #DFE9E2; border-radius: 14px;
        padding: 18px 20px; box-shadow: 0 5px 16px rgba(18,55,42,.05);
        height: 138px; box-sizing: border-box;
        display: flex; flex-direction: column; justify-content: center;
      }
      [data-testid="stMetricLabel"] { color: #64746F; }
      [data-testid="stMetricValue"] { color: #16302B; }
      .dashboard-title { font-size: 2.15rem; font-weight: 750; color: #16302B; }
      .dashboard-subtitle { color: #6B7C78; margin-top: -8px; margin-bottom: 20px; }
      .section-note {
        background: #EDF7F0; border-left: 4px solid #54B435; border-radius: 8px;
        padding: 11px 14px; color: #315548; margin: 6px 0 14px;
      }
      .status-chip {
        display: inline-block; border-radius: 999px; padding: 4px 10px;
        background: #E7F5EB; color: #1F7A3D; font-size: .82rem; font-weight: 650;
      }
      h1, h2, h3 { color: #16302B; }
      div[data-testid="stPlotlyChart"] {
        background: white; border: 1px solid #E2EAE5; border-radius: 14px;
        padding: 8px; box-shadow: 0 5px 16px rgba(18,55,42,.04);
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_bytes(
    file_bytes: bytes, schema_version: int
) -> tuple[pd.DataFrame, dict[str, object]]:
    _ = schema_version
    return load_kpi_data(BytesIO(file_bytes))


@st.cache_data(show_spinner=False, ttl=300)
def fetch_google_bytes(
    google_sheet_url: str, refresh_token: int
) -> tuple[bytes, str]:
    _ = refresh_token
    content = download_google_sheet(google_sheet_url)
    synced_at = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S")
    return content, synced_at


def percent(value: float | None) -> str:
    return "—" if value is None or pd.isna(value) else f"{value:.1%}"


def format_axis(fig: go.Figure, height: int = 390) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=28, r=24, t=55, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial, Microsoft YaHei, sans-serif", color=INK),
        legend_title_text="",
        hoverlabel=dict(bgcolor="white", font_color=INK),
    )
    fig.update_xaxes(gridcolor="#E9EFEB", zeroline=False)
    fig.update_yaxes(gridcolor="#E9EFEB", zeroline=False)
    return fig


def empty_chart(message: str) -> None:
    st.info(message, icon="ℹ️")


def format_kpi_value(value: float, metric_type: str) -> str:
    if pd.isna(value):
        return "—"
    if metric_type == "rate":
        return f"{value:.1%}"
    return f"{value:g}"


def build_raw_monthly_table(data: pd.DataFrame) -> pd.DataFrame:
    month_columns = sorted(data["Month"].drop_duplicates())
    display_source = data.copy()
    display_source["DisplayValue"] = display_source.apply(
        lambda row: (
            row["Reason"]
            if row["MetricType"] == "reason"
            else format_kpi_value(row["Value"], row["MetricType"])
        ),
        axis=1,
    )
    table = (
        display_source.pivot_table(
            index=["Job", "Name", "KPI", "KPIGroup", "MetricType"],
            columns="Month",
            values="DisplayValue",
            aggfunc="first",
        )
        .reset_index()
        .sort_values(["Name", "KPI"])
        .reset_index(drop=True)
    )
    table = table.rename(
        columns={
            "Job": "职位",
            "Name": "员工",
            "KPI": "二级KPI",
            "KPIGroup": "KPI分类",
            "MetricType": "数据类型",
            **{month: month.strftime("%Y/%m") for month in month_columns},
        }
    )
    table["数据类型"] = table["数据类型"].map(
        {
            "rate": "比率",
            "count": "次数",
            "duration": "周期",
            "reason": "原因",
        }
    ).fillna(table["数据类型"])
    return table


def build_reason_table(data: pd.DataFrame) -> pd.DataFrame:
    reasons = data[
        data["MetricType"].eq("reason") & data["Reason"].ne("")
    ][["Month", "Job", "Name", "Reason"]].copy()
    if reasons.empty:
        return pd.DataFrame()

    counts = (
        data[data["MetricType"].eq("count")]
        .groupby(["Month", "Job", "Name"], as_index=False)
        .agg(Count=("Value", "sum"))
    )
    reasons = reasons.merge(counts, on=["Month", "Job", "Name"], how="left")
    reasons["Month"] = reasons["Month"].dt.strftime("%Y/%m")
    reasons["Count"] = reasons["Count"].fillna(0).map(lambda value: f"{value:g}")
    return (
        reasons.rename(
            columns={
                "Month": "月份",
                "Job": "职位",
                "Name": "员工",
                "Count": "未准时次数",
                "Reason": "未准时提交原因",
            }
        )
        [["月份", "职位", "员工", "未准时次数", "未准时提交原因"]]
        .sort_values(["月份", "职位", "员工"], ascending=[False, True, True])
        .reset_index(drop=True)
    )


def build_attention_table(latest: pd.DataFrame) -> pd.DataFrame:
    attention = latest[latest["Target"].notna()].copy()
    attention = attention[
        (
            attention["MetricType"].eq("count")
            & attention["Value"].gt(attention["Target"])
        )
        | (
            ~attention["MetricType"].eq("count")
            & attention["Value"].lt(attention["Target"])
        )
    ].copy()
    if attention.empty:
        return pd.DataFrame()

    attention["Severity"] = attention.apply(
        lambda row: (
            row["Value"] - row["Target"]
            if row["MetricType"] == "count"
            else row["Target"] - row["Value"]
        ),
        axis=1,
    )
    attention["实际值"] = attention.apply(
        lambda row: format_kpi_value(row["Value"], row["MetricType"]), axis=1
    )
    attention["目标值"] = attention.apply(
        lambda row: format_kpi_value(row["Target"], row["MetricType"]), axis=1
    )
    attention["差距"] = attention.apply(
        lambda row: (
            f"超出 {row['Severity']:g} 次"
            if row["MetricType"] == "count"
            else f"低于目标 {row['Severity']:.1%}"
        ),
        axis=1,
    )
    attention["建议动作"] = attention["KPIGroup"].map(
        {
            "异常次数": "核对逾期原因、责任人和补救日期",
            "RFT": "复盘首轮失败原因并完善交付前检查清单",
            "准时交付": "检查排期、前置依赖和交付节点",
        }
    ).fillna("确认未达标原因并制定改善措施")
    reason_lookup = (
        latest[
            latest["MetricType"].eq("reason") & latest["Reason"].ne("")
        ]
        .groupby(["Job", "Name"])["Reason"]
        .agg("；".join)
        .to_dict()
    )
    attention["未准时提交原因"] = attention.apply(
        lambda row: (
            reason_lookup.get((row["Job"], row["Name"]), "未填写")
            if row["MetricType"] == "count"
            else "—"
        ),
        axis=1,
    )
    return (
        attention.sort_values("Severity", ascending=False)
        .rename(columns={"Job": "职位", "Name": "员工", "KPI": "二级KPI"})
        [
            [
                "职位",
                "员工",
                "二级KPI",
                "实际值",
                "目标值",
                "差距",
                "未准时提交原因",
                "建议动作",
            ]
        ]
        .reset_index(drop=True)
    )


with st.sidebar:
    st.markdown("## 数据控制台")
    data_source = st.radio(
        "数据来源",
        ["Google 表格自动同步", "上传 Excel", "内置模板"],
    )

    uploaded = None
    if data_source == "Google 表格自动同步":
        st.caption("默认每 5 分钟重新获取一次公开 Google 表格。")
        st.link_button("打开 Google 表格", GOOGLE_SHEET_URL, width="stretch")
        if "google_refresh_token" not in st.session_state:
            st.session_state["google_refresh_token"] = 0
        if st.button("立即刷新 Google 数据", width="stretch"):
            st.session_state["google_refresh_token"] += 1
            st.rerun()
    elif data_source == "上传 Excel":
        st.caption("上传整份 Excel，系统会自动读取“工作表2”（或第二张工作表）。")
        uploaded = st.file_uploader("上传 KPI Excel", type=["xlsx", "xlsm"])
    else:
        st.caption("使用项目内保存的 Excel 数据。")

    with SAMPLE_FILE.open("rb") as sample:
        st.download_button(
            "下载当前 Excel 模板",
            data=sample.read(),
            file_name="DCKPI Dashboard.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
        )

try:
    if data_source == "Google 表格自动同步":
        file_bytes, synced_at = fetch_google_bytes(
            GOOGLE_SHEET_URL, st.session_state["google_refresh_token"]
        )
        source_label = "Google 表格"
    elif data_source == "上传 Excel":
        if uploaded is None:
            st.info("请上传 Excel 文件。")
            st.stop()
        file_bytes = uploaded.getvalue()
        synced_at = datetime.now(ZoneInfo("Asia/Shanghai")).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        source_label = "上传的 Excel"
    else:
        file_bytes = SAMPLE_FILE.read_bytes()
        synced_at = "项目内置文件"
        source_label = "内置模板"

    data, source_info = load_bytes(file_bytes, DATA_SCHEMA_VERSION)
    source_info["source_label"] = source_label
    source_info["synced_at"] = synced_at
except Exception as exc:
    st.error(f"数据读取失败：{exc}")
    st.stop()

with st.sidebar:
    if source_info["source_label"] == "Google 表格":
        st.success(f"Google 表格已连接\n\n最近同步：{source_info['synced_at']}")
    else:
        st.info(f"当前数据源：{source_info['source_label']}")

all_months = sorted(data["Month"].drop_duplicates().tolist())
latest_available = max(all_months)

with st.sidebar:
    st.divider()
    st.markdown("### 筛选范围")
    period_mode = st.radio(
        "统计窗口",
        ["最近12个月", "本年累计（YTD）", "自定义"],
        horizontal=False,
    )

    if period_mode == "本年累计（YTD）":
        start_month = pd.Timestamp(latest_available.year, 1, 1)
        end_month = latest_available
    elif period_mode == "最近12个月":
        start_month = latest_available - pd.DateOffset(months=11)
        end_month = latest_available
    else:
        month_labels = [month.strftime("%Y/%m") for month in all_months]
        if len(month_labels) > 1:
            start_label, end_label = st.select_slider(
                "月份",
                options=month_labels,
                value=(month_labels[0], month_labels[-1]),
            )
        else:
            start_label = end_label = month_labels[0]
            st.caption(f"月份：{start_label}")
        start_month = pd.to_datetime(start_label, format="%Y/%m")
        end_month = pd.to_datetime(end_label, format="%Y/%m")

    job_options = sorted(data["Job"].unique())
    selected_jobs = st.multiselect("职位", job_options, default=job_options)
    available_names = sorted(data[data["Job"].isin(selected_jobs)]["Name"].unique())
    selected_names = st.multiselect(
        "员工", available_names, default=available_names
    )

filtered = data[
    data["Month"].between(start_month, end_month)
    & data["Job"].isin(selected_jobs)
    & data["Name"].isin(selected_names)
].copy()

if filtered.empty:
    st.warning("当前筛选没有数据，请调整月份、职位或员工。")
    st.stop()

latest_month = filtered["Month"].max()
period_start = filtered["Month"].min()
period_end = filtered["Month"].max()
period_label = (
    f"{period_start:%Y/%m}"
    if period_start == period_end
    else f"{period_start:%Y/%m}–{period_end:%Y/%m}"
)
previous_months = sorted(filtered.loc[filtered["Month"] < latest_month, "Month"].unique())
previous_month = previous_months[-1] if previous_months else None
latest_data = filtered[filtered["Month"].eq(latest_month)]
previous_data = (
    filtered[filtered["Month"].eq(previous_month)]
    if previous_month is not None
    else pd.DataFrame(columns=filtered.columns)
)

rate_latest = latest_data[latest_data["MetricType"].eq("rate")]
rft_latest = rate_latest[rate_latest["KPIGroup"].eq("RFT")]
ontime_latest = rate_latest[rate_latest["KPIGroup"].eq("准时交付")]
count_latest = latest_data[latest_data["MetricType"].eq("count")]

target_rows = latest_data[latest_data["Target"].notna()]
target_achievement = (
    float(target_rows["TargetMet"].astype(float).mean()) if not target_rows.empty else None
)
rft_value = float(rft_latest["Value"].mean()) if not rft_latest.empty else None
ontime_value = float(ontime_latest["Value"].mean()) if not ontime_latest.empty else None
overdue_value = float(count_latest["Value"].sum()) if not count_latest.empty else 0.0

previous_rft = previous_data[
    previous_data["MetricType"].eq("rate") & previous_data["KPIGroup"].eq("RFT")
]["Value"].mean()
rft_delta = (
    f"{rft_value - previous_rft:+.1%}"
    if rft_value is not None and pd.notna(previous_rft)
    else None
)

st.markdown(
    '<div class="dashboard-title">ZX R&D KPI Dashboard</div>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<div class="dashboard-subtitle">研发团队绩效监控 · 数据截止 {latest_month:%Y年%m月} · '
    f'{len(selected_jobs)} 个职位 / {filtered["Name"].nunique()} 名员工</div>',
    unsafe_allow_html=True,
)

metric_cols = st.columns(5)
metric_cols[0].metric("RFT 一次通过率", percent(rft_value), rft_delta)
metric_cols[1].metric("准时交付率", percent(ontime_value))
metric_cols[2].metric("KPI 达标率", percent(target_achievement))
metric_cols[3].metric("未准时提交次数", f"{overdue_value:g}")
metric_cols[4].metric("在册员工", f"{filtered['Name'].nunique()} 人")

tabs = st.tabs(["管理总览", "岗位专项", "绩效明细", "数据说明"])

with tabs[0]:
    st.markdown("### 团队 KPI 趋势")
    st.markdown(
        '<div class="section-note">比率类指标按月取平均；当前 Excel 没有分子/分母明细，因此团队汇总采用等权平均。</div>',
        unsafe_allow_html=True,
    )

    trend = (
        filtered[filtered["MetricType"].eq("rate")]
        .groupby(["Month", "Job"], as_index=False)
        .agg(Value=("Value", "mean"))
    )
    overall = (
        filtered[filtered["MetricType"].eq("rate")]
        .groupby("Month", as_index=False)
        .agg(Value=("Value", "mean"))
    )
    fig_trend = px.line(
        trend,
        x="Month",
        y="Value",
        color="Job",
        line_dash="Job",
        symbol="Job",
        markers=True,
        color_discrete_sequence=[GREEN, BLUE, YELLOW, "#9B51E0"],
        labels={"Value": "平均达成率", "Month": "月份", "Job": "职位"},
        title="各职位比率类 KPI 月度表现",
    )
    fig_trend.update_traces(
        line=dict(width=3),
        marker=dict(size=8),
        hovertemplate=(
            "<b>职位：%{fullData.name}</b><br>"
            "月份：%{x|%Y/%m}<br>"
            "平均达成率：%{y:.1%}<extra></extra>"
        ),
    )
    fig_trend.add_trace(
        go.Scatter(
            x=overall["Month"],
            y=overall["Value"],
            name="团队整体",
            mode="lines+markers",
            line=dict(color=INK, width=4, dash="dot"),
            marker=dict(size=9, symbol="diamond"),
            hovertemplate=(
                "<b>团队整体</b><br>"
                "月份：%{x|%Y/%m}<br>"
                "平均达成率：%{y:.1%}<extra></extra>"
            ),
        )
    )
    fig_trend.update_xaxes(tickformat="%Y/%m")
    fig_trend.update_yaxes(
        tickformat=".0%",
        tick0=0.5,
        dtick=0.1,
        range=[0.5, max(1.02, trend["Value"].max() * 1.02)],
    )
    st.plotly_chart(
        format_axis(fig_trend, 460),
        config={"displayModeBar": False},
    )

    left, right = st.columns([1.05, 1])
    with left:
        employee_source = filtered[
            filtered["MetricType"].eq("rate") & filtered["Target"].notna()
        ]
        employee = (
            employee_source
            .assign(
                Attainment=lambda frame: frame["Value"]
                / frame["Target"].replace(0, pd.NA)
            )
            .groupby(["Job", "Name"], as_index=False)
            .agg(
                Attainment=("Attainment", "mean"),
                AverageRate=("Value", "mean"),
                Months=("Month", "nunique"),
            )
            .sort_values("Attainment")
        )
        employee["Status"] = employee["Attainment"].ge(1).map(
            {True: "达标", False: "需关注"}
        )
        fig_employee = px.bar(
            employee,
            x="Attainment",
            y="Name",
            color="Status",
            orientation="h",
            text=employee["Attainment"].map(lambda value: f"{value:.0%}"),
            custom_data=["Job", "Months", "AverageRate"],
            color_discrete_map={"达标": GREEN, "需关注": RED},
            title=f"个人目标达成指数 · {period_label}",
            labels={
                "Attainment": "目标达成指数",
                "Name": "员工",
                "Job": "职位",
                "Months": "有数据月份",
                "AverageRate": "期间平均达成率",
                "Status": "状态",
            },
        )
        fig_employee.update_traces(
            hovertemplate=(
                "<b>员工：%{y}</b><br>"
                "职位：%{customdata[0]}<br>"
                "有数据月份：%{customdata[1]}<br>"
                "期间平均达成率：%{customdata[2]:.1%}<br>"
                "目标达成指数：%{x:.1%}<br>"
                "状态：%{fullData.name}<extra></extra>"
            )
        )
        fig_employee.add_vline(x=1, line_dash="dash", line_color=INK)
        fig_employee.update_xaxes(tickformat=".0%")
        st.plotly_chart(
            format_axis(fig_employee, 390),
            config={"displayModeBar": False},
        )

    with right:
        heat_source = filtered[filtered["MetricType"].eq("rate")]
        heat = heat_source.pivot_table(
            index="Name", columns="KPI", values="Value", aggfunc="mean"
        )
        if not heat.empty:
            z = heat.to_numpy() * 100
            text = [
                ["" if pd.isna(value) else f"{value:.0f}%" for value in row]
                for row in z
            ]
            fig_heat = go.Figure(
                go.Heatmap(
                    z=z,
                    x=heat.columns,
                    y=heat.index,
                    text=text,
                    texttemplate="%{text}",
                    colorscale=[
                        [0, "#FDE7E7"],
                        [0.80, "#F7E7A7"],
                        [0.95, "#D7F0D1"],
                        [1, GREEN],
                    ],
                    zmin=50,
                    zmax=100,
                    colorbar=dict(title="%"),
                    hovertemplate=(
                        "<b>员工：%{y}</b><br>"
                        "KPI：%{x}<br>"
                        "期间平均：%{z:.1f}%<extra></extra>"
                    ),
                )
            )
            fig_heat.update_layout(title=f"KPI 期间平均热力图 · {period_label}")
            st.plotly_chart(
                format_axis(fig_heat, 390),
                config={"displayModeBar": False},
            )
        else:
            empty_chart("当前范围没有比率类 KPI。")

    st.markdown("### 未准时提交趋势")
    exception = filtered[filtered["MetricType"].eq("count")]
    if not exception.empty:
        exception_monthly = (
            exception.groupby(["Month", "Job", "Name"], as_index=False)
            .agg(Value=("Value", "sum"))
            .sort_values("Month")
        )
        reason_monthly = (
            filtered[
                filtered["MetricType"].eq("reason") & filtered["Reason"].ne("")
            ]
            .groupby(["Month", "Job", "Name"], as_index=False)
            .agg(Reason=("Reason", "；".join))
        )
        exception_monthly = exception_monthly.merge(
            reason_monthly, on=["Month", "Job", "Name"], how="left"
        )
        exception_monthly["Reason"] = exception_monthly["Reason"].fillna("未填写")
        fig_exception = px.bar(
            exception_monthly,
            x="Month",
            y="Value",
            color="Name",
            custom_data=["Job", "Reason"],
            barmode="group",
            color_discrete_sequence=px.colors.qualitative.Set2,
            title="逾期/未准时提交次数（越低越好）",
            labels={"Value": "次数", "Month": "月份", "Name": "员工"},
        )
        fig_exception.update_traces(
            hovertemplate=(
                "<b>员工：%{fullData.name}</b><br>"
                "职位：%{customdata[0]}<br>"
                "月份：%{x|%Y/%m}<br>"
                "未准时提交：%{y:.0f} 次<br>"
                "原因：%{customdata[1]}<extra></extra>"
            )
        )
        fig_exception.update_xaxes(tickformat="%Y/%m")
        fig_exception.update_yaxes(dtick=1)
        st.plotly_chart(
            format_axis(fig_exception, 350),
            config={"displayModeBar": False},
        )
    else:
        empty_chart("当前范围没有“次数”类异常指标。")

    st.markdown(f"### 未准时提交原因 · {period_label}")
    reason_table = build_reason_table(filtered)
    if reason_table.empty:
        st.info("当前筛选范围没有填写未准时提交原因。", icon="ℹ️")
    else:
        st.dataframe(
            reason_table,
            width="stretch",
            hide_index=True,
            height=min(380, 38 * (len(reason_table) + 1)),
        )

    st.markdown(f"### 最新需关注清单 · {latest_month:%Y/%m}")
    attention_table = build_attention_table(latest_data)
    if attention_table.empty:
        st.success("最新月份所有有目标的 KPI 均已达标。", icon="✅")
    else:
        st.caption("自动汇总低于目标的比率 KPI，以及大于目标值的次数类 KPI。")
        st.dataframe(
            attention_table,
            width="stretch",
            hide_index=True,
            height=min(420, 38 * (len(attention_table) + 1)),
        )

with tabs[1]:
    modelist_col, designer_col = st.columns([1.25, 0.75])

    with modelist_col:
        st.markdown("### Modelist：TP Qualification")
        modelist = latest_data[
            latest_data["Job"].str.contains("Modelist", case=False, na=False)
            & latest_data["MetricType"].eq("rate")
        ]
        modelist_pivot = modelist.pivot_table(
            index="Name", columns="KPI", values="Value", aggfunc="mean"
        )

        def find_column(keyword: str) -> str | None:
            return next(
                (column for column in modelist_pivot.columns if keyword in column.upper()),
                None,
            )

        pap_col = find_column("PAP")
        tf_col = find_column("TF")
        bom_col = find_column("BOM")
        if pap_col and tf_col and bom_col and not modelist_pivot.empty:
            bubble = modelist_pivot[[pap_col, tf_col, bom_col]].dropna().reset_index()
            fig_bubble = go.Figure()
            fig_bubble.add_trace(
                go.Scatter(
                    x=bubble[tf_col] * 100,
                    y=bubble[pap_col] * 100,
                    mode="markers+text",
                    text=bubble["Name"],
                    textposition="top center",
                    marker=dict(
                        size=(bubble[bom_col].clip(lower=0) * 38 + 18),
                        color=bubble[bom_col] * 100,
                        colorscale=[
                            [0, RED],
                            [0.65, YELLOW],
                            [0.95, GREEN],
                            [1, DARK_GREEN],
                        ],
                        cmin=50,
                        cmax=100,
                        showscale=True,
                        colorbar=dict(title="BOM<br>通过率"),
                        line=dict(width=2, color="white"),
                    ),
                    customdata=bubble[[bom_col]].to_numpy(),
                    hovertemplate=(
                        "<b>员工：%{text}</b><br>"
                        "TF 一次通过率：%{x:.1f}%<br>"
                        "PAP 一次通过率：%{y:.1f}%<br>"
                        "BOM 一次通过率：%{customdata[0]:.1%}<extra></extra>"
                    ),
                )
            )
            fig_bubble.update_layout(
                title=f"PAP × TF × BOM · {latest_month:%Y/%m}",
                xaxis_title="TF 一次通过率",
                yaxis_title="PAP 一次通过率",
            )
            fig_bubble.update_xaxes(range=[50, 102], ticksuffix="%")
            fig_bubble.update_yaxes(range=[50, 102], ticksuffix="%")
            st.plotly_chart(
                format_axis(fig_bubble, 470),
                config={"displayModeBar": False},
            )
        else:
            empty_chart("需要同时存在 PAP、TF、BOM 三项数据才能生成 Modelist 气泡图。")

    with designer_col:
        st.markdown("### Designer：3D 品质")
        designer = latest_data[
            latest_data["Job"].str.contains("Designer", case=False, na=False)
            & latest_data["MetricType"].eq("rate")
        ]
        if not designer.empty:
            pass_rate = float(designer["Value"].mean())
            pass_rate = min(max(pass_rate, 0), 1)
            fig_donut = go.Figure(
                go.Pie(
                    labels=["合格", "不合格"],
                    values=[pass_rate, 1 - pass_rate],
                    hole=0.55,
                    marker_colors=[GREEN, YELLOW],
                    textinfo="label+percent",
                    hovertemplate="<b>%{label}</b><br>占比：%{percent}<extra></extra>",
                    sort=False,
                    direction="clockwise",
                )
            )
            fig_donut.update_layout(
                title=f"3D 交付准确率 · {latest_month:%Y/%m}",
                showlegend=False,
                annotations=[
                    dict(
                        text=f"{pass_rate:.0%}",
                        x=0.5,
                        y=0.5,
                        font_size=28,
                        showarrow=False,
                    )
                ],
            )
            st.plotly_chart(
                format_axis(fig_donut, 470),
                config={"displayModeBar": False},
            )
        else:
            empty_chart("当前范围没有 Designer 的比率数据。")

    st.markdown("### PIS 与 IE 月度明细")
    focus = filtered[
        filtered["Job"].str.contains("PIS|IE", case=False, regex=True, na=False)
        & filtered["MetricType"].eq("rate")
    ]
    if not focus.empty:
        fig_focus = px.line(
            focus,
            x="Month",
            y="Value",
            color="KPI",
            facet_row="Job",
            markers=True,
            color_discrete_sequence=px.colors.qualitative.Safe,
            title="PIS / IE KPI 趋势",
            labels={
                "Value": "达成率",
                "Month": "月份",
                "KPI": "二级KPI",
                "Job": "职位",
            },
        )
        fig_focus.update_traces(
            hovertemplate=(
                "<b>KPI：%{fullData.name}</b><br>"
                "月份：%{x|%Y/%m}<br>"
                "达成率：%{y:.1%}<extra></extra>"
            )
        )
        fig_focus.update_xaxes(tickformat="%Y/%m")
        fig_focus.update_yaxes(tickformat=".0%", range=[0, 1.08])
        fig_focus.for_each_annotation(lambda annotation: annotation.update(text=annotation.text.split("=")[-1]))
        st.plotly_chart(
            format_axis(fig_focus, 520),
            config={"displayModeBar": False},
        )

    lead_time = filtered[filtered["KPIGroup"].eq("L/T")]
    if lead_time.empty:
        st.warning(
            "L/T（Lead Time / 周期）暂未显示：当前工作表2没有周期或时长类 KPI。"
            "后续增加名称含“L/T、Lead Time、周期、时长或天数”的行后，系统会自动识别。",
            icon="⏱️",
        )

with tabs[2]:
    st.markdown(f"### 工作表2原始月度数据 · {period_label}")
    st.caption(
        "每个 KPI 保留为独立行；比率、次数和原因不会混在一起。"
        "例如 Bethy 的“样品一次通过率”“未准时提交的次数”和原因会分别展示。"
    )
    raw_monthly = build_raw_monthly_table(filtered)
    st.dataframe(
        raw_monthly,
        width="stretch",
        hide_index=True,
        height=min(620, 38 * (len(raw_monthly) + 1)),
    )
    st.download_button(
        "下载当前筛选的原始月度数据",
        data=raw_monthly.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"ZX_KPI原始数据_{period_start:%Y%m}_{period_end:%Y%m}.csv",
        mime="text/csv",
    )

    st.markdown("### 最新月份与期间表现")
    detail = period_detail(filtered)
    display = detail.copy()
    if not display.empty:
        for column in ("Latest", "PeriodAverage", "Target"):
            display[column] = display.apply(
                lambda row: (
                    f"{row[column]:g}"
                    if row.get("MetricType") in ("count", "duration")
                    and pd.notna(row[column])
                    else f"{row[column]:.1%}"
                    if pd.notna(row[column])
                    else "—"
                ),
                axis=1,
            )
        display = display.rename(
            columns={
                "Job": "职位",
                "Name": "员工",
                "KPI": "二级KPI",
                "KPIGroup": "分类",
                "Latest": f"{latest_month:%Y/%m}",
                "PeriodAverage": "期间平均",
                "Target": "目标",
                "Months": "有数据月份",
                "Status": "状态",
            }
        )
        display = display[
            [
                "职位",
                "员工",
                "二级KPI",
                "分类",
                f"{latest_month:%Y/%m}",
                "期间平均",
                "目标",
                "状态",
                "有数据月份",
            ]
        ]
        st.dataframe(
            display,
            width="stretch",
            hide_index=True,
            height=min(620, 38 * (len(display) + 1)),
        )

    with st.expander("查看标准化后的明细数据"):
        raw_display = filtered.copy()
        raw_display["Month"] = raw_display["Month"].dt.strftime("%Y/%m")
        raw_display["Value"] = raw_display.apply(
            lambda row: (
                row["Reason"]
                if row["MetricType"] == "reason"
                else
                f"{row['Value']:.1%}"
                if row["MetricType"] == "rate"
                else f"{row['Value']:g}"
            ),
            axis=1,
        )
        raw_display = raw_display.rename(
            columns={
                "Month": "月份",
                "Job": "职位",
                "Name": "员工",
                "KPI": "二级KPI",
                "KPIGroup": "KPI分类",
                "MetricType": "数据类型",
                "Value": "数值",
            }
        )
        raw_display["数据类型"] = raw_display["数据类型"].map(
            {
                "rate": "比率",
                "count": "次数",
                "duration": "周期",
                "reason": "原因",
            }
        ).fillna(raw_display["数据类型"])
        st.dataframe(
            raw_display[
                ["月份", "职位", "员工", "二级KPI", "KPI分类", "数据类型", "数值"]
            ],
            width="stretch",
            hide_index=True,
        )

with tabs[3]:
    st.markdown("### Excel 更新规则")
    st.markdown(
        """
        1. 保留前三列：`Job`、`Name`、`2级KPI`。
        2. 从第 4 列开始，每一列代表一个月份；建议直接使用 Excel 日期并显示为 `yyyy/m`。
        3. 比率可填写 `95%`、`0.95` 或 `95`，系统会自动统一。
        4. “未准时提交的次数”等计数 KPI 直接填写整数。
        5. 原因使用独立行“未准时提交的原因”，在对应月份填写文字。
        6. 上传整份 Excel 即可，系统优先读取名为“工作表2”的工作表。
        """
    )
    quality_cols = st.columns(5)
    quality_cols[0].metric("数据来源", str(source_info["source_label"]))
    quality_cols[1].metric("职位", str(source_info["job_count"]))
    quality_cols[2].metric("员工", str(source_info["employee_count"]))
    quality_cols[3].metric("月份", str(source_info["month_count"]))
    quality_cols[4].metric("有效数据点", str(source_info["data_points"]))
    st.caption(
        f"读取工作表：{source_info['sheet_name']} · "
        f"最近同步：{source_info['synced_at']}"
    )
    st.caption(f"其中包含 {source_info['reason_count']} 条未准时提交原因记录。")

    if source_info["invalid_numeric"]:
        st.warning(f"发现 {source_info['invalid_numeric']} 个非数字 KPI 值，已跳过。")
    if source_info["duplicate_count"]:
        st.warning(
            f"发现 {source_info['duplicate_count']} 条重复的“职位+员工+KPI+月份”记录，"
            "图表会按平均值汇总。"
        )
    st.markdown(
        '<span class="status-chip">当前数据结构校验通过</span>',
        unsafe_allow_html=True,
    )

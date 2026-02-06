import streamlit as st
import pandas as pd
import plotly.express as px
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# --- 0. 页面全局设置 ---
st.set_page_config(
    page_title="中兴手套开发中心Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. CSS 美化 ---
st.markdown("""
<style>
    .stApp { background-color: #f4f5f7; font-family: 'PingFang SC', sans-serif; }
    h1, h2, h3 { color: #172b4d; font-weight: 700; }
    div[data-testid="stMetric"] {
        background-color: #ffffff; padding: 15px; border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05); border-left: 5px solid #0052cc;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #ffffff; border-radius: 4px; font-weight: 600; }
    .stTabs [aria-selected="true"] { background-color: #deebff; color: #0052cc; }
    
    /* 报告文本样式 */
    .report-text { font-size: 16px; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)

# --- 2. 核心逻辑层 ---

@st.cache_data(ttl=300)
def load_business_data(file_path, simulation_date=None):
    try:
        df = pd.read_excel(file_path, engine="openpyxl")
    except:
        return None

    df.columns = df.columns.str.strip()
    
    date_cols = ['下单日期', '要求交期', '发货日期', '技术确认日期']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    today = pd.to_datetime(simulation_date) if simulation_date else pd.to_datetime(datetime.date.today())

    # 1. 完工日期
    if '发货日期' in df.columns and '技术确认日期' in df.columns:
        df['完工日期'] = df['发货日期'].fillna(df['技术确认日期'])
    else:
        df['完工日期'] = pd.NaT

    # 2. 状态判定
    def evaluate_status(row):
        deadline = row.get('要求交期')
        done_date = row.get('完工日期')
        
        if pd.isnull(deadline): return "⚪ 未知"
        
        is_completed = pd.notnull(done_date) and (done_date <= today)
        
        if is_completed:
            if done_date > deadline:
                return "⚠️ 逾期交付 (历史)"
            else:
                return "✅ 按时交付"
        else:
            days_left = (deadline - today).days
            if days_left < 0:
                return "🔴 严重逾期 (进行中)"
            elif days_left <= 3:
                return "🟠 紧急 (3天内)"
            else:
                return "🔵 正常进行"

    df['业务状态'] = df.apply(evaluate_status, axis=1)
    
    # 3. 辅助计算 (剩余天数)
    def calc_days_gap(row):
        deadline = row.get('要求交期')
        done_date = row.get('完工日期')
        
        if "历史" in row['业务状态']:
            return (done_date - deadline).days
        else:
            return (deadline - today).days

    df['时间差指标'] = df.apply(calc_days_gap, axis=1)

    # 4. 填充
    df['寄出总数量'] = df['寄出总数量'].fillna(0)
    for c in ['客户', '款式', '业务员', '设计员']:
        if c in df.columns: df[c] = df[c].fillna("未知")

    return df, today

# --- 自动化归因分析报告生成函数 ---
def generate_insight_report(df, today):
    risk_df = df[df['业务状态'].isin(["🔴 严重逾期 (进行中)", "🟠 紧急 (3天内)", "⚠️ 逾期交付 (历史)"])]
    today_str = today.strftime("%Y-%m-%d")
    total_count = len(df)
    risk_count = len(risk_df)
    
    if risk_count == 0:
        return True, f"""
        ### ✅ {today_str} 生产运营日报
        **状态：优**
        今日生产线运行平稳，共监控 **{total_count}** 个订单，全线无逾期、无临期风险。
        """
    else:
        top_cust = risk_df['客户'].value_counts().idxmax()
        top_cust_count = risk_df['客户'].value_counts().max()
        top_sales = risk_df['业务员'].value_counts().idxmax()
        
        return False, f"""
        ### ⚠️ {today_str} 生产运营日报 (自动生成)
        
        当前监控订单总数 **{total_count}** 单，发现异常/风险单 **{risk_count}** 单。
        
        #### 📊 核心归因分析：
        1.  **最大风险源 (客户)**：**{top_cust}**。该客户关联了 **{top_cust_count}** 个异常订单，占总异常的 {top_cust_count/risk_count*100:.0f}%。
            * *行动建议：请业务部优先与 {top_cust} 确认船期或修改交期。*
        2.  **内部负荷分析**：业务员 **{top_sales}** 名下的异常单据最多。
            * *行动建议：建议主管检查该业务员是否存在单据积压或沟通受阻情况。*
        """

# --- 3. 页面渲染 ---

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/factory.png", width=60)
    st.markdown("### 中兴开发中心Dashboard")
    
    real_today = datetime.date.today()
    sim_date = st.date_input("基准日期", value=real_today)
    
    st.markdown("---")
    uploaded_file = st.file_uploader("📂 数据源 (Excel)", type=["xlsx"])
    
    if uploaded_file:
        df, current_date = load_business_data(uploaded_file, sim_date)
    else:
        try:
            df, current_date = load_business_data("样品传递单.xlsx", sim_date)
        except:
            df = None

if df is not None:
    
    # === KPI ===
    current_overdue = len(df[df['业务状态'] == "🔴 严重逾期 (进行中)"])
    current_urgent = len(df[df['业务状态'] == "🟠 紧急 (3天内)"])
    history_bad = len(df[df['业务状态'] == "⚠️ 逾期交付 (历史)"])
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🚨 需立即干预", f"{current_overdue} 单", "已逾期", delta_color="inverse")
    col2.metric("🟠 3日内临期", f"{current_urgent} 单", "即将逾期", delta_color="inverse")
    col3.metric("⚠️ 历史逾期", f"{history_bad} 单", "已完工", delta_color="off")
    
    total_orders = len(df)
    total_issues = current_overdue + history_bad
    rate = (total_issues / total_orders * 100) if total_orders > 0 else 0
    col4.metric("履约异常率", f"{rate:.1f}%", f"共 {total_issues} 异常")

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs([
        "1.风险管控 (Risk Log)", 
        "2.技术员绩效", 
        "3.智能洞察", 
        "4.剩余(1-3天)订单"
    ])

    # === Tab 1: 风险与问题管控 ===
    with tab1:
        st.markdown("### 问题订单追踪")
        problem_mask = df['业务状态'].isin(["🔴 严重逾期 (进行中)", "🟠 紧急 (3天内)", "⚠️ 逾期交付 (历史)"])
        problem_df = df[problem_mask].copy()
        
        status_priority = {
            "🔴 严重逾期 (进行中)": 1,
            "🟠 紧急 (3天内)": 2,
            "⚠️ 逾期交付 (历史)": 3
        }
        problem_df['priority'] = problem_df['业务状态'].map(status_priority)
        display_df = problem_df.sort_values(['priority', '时间差指标'])
        
        if display_df.empty:
            st.success("🎉 当前无风险订单。")
        else:
            st.dataframe(
                display_df[['业务状态', '样品传递单号', '客户', '款式', '设计员', '要求交期', '时间差指标']],
                column_config={
                    "时间差指标": st.column_config.NumberColumn("剩余/超期天数", format="%d 天"),
                    "要求交期": st.column_config.DateColumn("要求交期", format="MM-DD"),
                },
                use_container_width=True,
                height=500
            )

    # === Tab 2: 绩效 (款式数 X轴) ===
    with tab2:
        st.markdown("### 🏆 技术效能矩阵")
        
        perf_df = df.groupby('设计员').agg(
            总接单量=('样品传递单号', 'nunique'),
            打样款式数=('款式', 'nunique')
        ).reset_index()
        
        finished_df = df[df['业务状态'].isin(["✅ 按时交付", "⚠️ 逾期交付 (历史)"])]
        
        if not finished_df.empty:
            tech_stats = finished_df.groupby('设计员').apply(
                lambda x: pd.Series({
                    '考核单量': len(x),
                    '及时单量': len(x[x['业务状态'] == "✅ 按时交付"])
                })
            ).reset_index()
            
            full_stats = pd.merge(perf_df, tech_stats, on='设计员', how='left').fillna(0)
            full_stats['及时率'] = (full_stats['及时单量'] / full_stats['考核单量'] * 100).round(1)
            full_stats = full_stats[full_stats['总接单量'] > 0] 

            fig_bubble = px.scatter(
                full_stats, x="打样款式数", y="及时率", size="总接单量", color="及时率",
                text="设计员", color_continuous_scale="RdYlGn", size_max=60,
                title="人员效能：开发款式数(X) vs 及时率(Y)",
                labels={"打样款式数": "开发款式 (款)", "及时率": "及时率 (%)"}
            )
            fig_bubble.add_hline(y=90, line_dash="dot", annotation_text="90% 及格")
            st.plotly_chart(fig_bubble, use_container_width=True)

    # === Tab 3: 智能洞察 (简报 + 原有图表) ===
    with tab3:
        st.markdown("### 业务洞察")
        
        # 1. 自动化报告 (简报)
        is_success, report_text = generate_insight_report(df, current_date)
        st.markdown("#### 每日自动归因简报")
        if is_success:
            st.success(report_text)
        else:
            st.warning(report_text)
            
        st.divider()

        # 2. 保留原有的图表分析 (重点保留)
        st.markdown("#### 📈 关键数据趋势")
        c1, c2 = st.columns(2)
        with c1:
            # 图表1：月度逾期率
            df['月'] = df['要求交期'].dt.to_period('M').astype(str)
            trend_df = df.groupby('月').apply(lambda x: (x['业务状态'].str.contains('逾期')).sum() / len(x) * 100).reset_index(name='逾期率')
            fig_trend = px.line(trend_df, x='月', y='逾期率', title="月度逾期率趋势 (%)", markers=True)
            st.plotly_chart(fig_trend, use_container_width=True)
        with c2:
            # 图表2：业务员逾期排行
            sales_delay = df[df['业务状态'].str.contains('逾期')].groupby('业务员').size().reset_index(name='单数').sort_values('单数', ascending=False).head(10)
            fig_sales = px.bar(sales_delay, x='单数', y='业务员', orientation='h', title="业务员逾期排行 (Top 10)")
            st.plotly_chart(fig_sales, use_container_width=True)
            
        st.divider()

        # 3. AI 预测部分
        train_df = df[df['业务状态'].isin(["✅ 按时交付", "⚠️ 逾期交付 (历史)"])].copy()
        pred_df = df[df['业务状态'].isin(["🔵 正常进行", "🟠 紧急 (3天内)"])].copy()
        if len(train_df) > 5 and len(pred_df) > 0:
             train_df['Is_Late'] = train_df['业务状态'].apply(lambda x: 1 if "逾期" in str(x) else 0)
             le = LabelEncoder()
             le.fit(pd.concat([train_df['客户'].astype(str), pred_df['客户'].astype(str)]).unique())
             train_df['C'] = le.transform(train_df['客户'].astype(str))
             pred_df['C'] = le.transform(pred_df['客户'].astype(str))
             
             model = RandomForestClassifier(n_estimators=50, random_state=42)
             model.fit(train_df[['C', '寄出总数量']], train_df['Is_Late'])
             pred_df['Risk'] = model.predict_proba(pred_df[['C', '寄出总数量']])[:, 1]
             
             st.markdown("#### 🤖 AI 风险预测 (Machine Learning)")
             st.dataframe(pred_df.sort_values('Risk', ascending=False)[['样品传递单号', '客户', 'Risk']].head(5), use_container_width=True)

    # === Tab 4: 剩余(1-3天)订单 ===
    with tab4:
        st.markdown("### 1-3日紧急订单 (Last Minute Rescue)")
        st.caption("🚨 **预警逻辑：** 筛选距离截止日期 **仅剩 1-3 天** 的订单。如果不在此期间完成，3天后它们将全部变成逾期单！这是最后的补救窗口。")

        rescue_mask = (df['时间差指标'] >= 1) & (df['时间差指标'] <= 3)
        rescue_mask = rescue_mask & (df['完工日期'].isna() | (df['完工日期'] > current_date))
        
        rescue_df = df[rescue_mask].copy()

        if not rescue_df.empty:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("3日内到期订单", f"{len(rescue_df)} 单", "必须优先排产", delta_color="inverse")
            with c2:
                most_urgent_day = rescue_df['时间差指标'].min()
                st.metric("最短剩余时间", f"{most_urgent_day} 天", "立即产出", delta_color="inverse")
            with c3:
                cust_count = rescue_df['客户'].nunique()
                st.metric("涉及客户数", f"{cust_count} 家", "需提前沟通")

            st.divider()

            c_chart, c_list = st.columns([1, 1])

            with c_chart:
                st.markdown("#### 倒计时分布")
                count_by_day = rescue_df['时间差指标'].value_counts().reset_index()
                count_by_day.columns = ['剩余天数', '单量']
                count_by_day['剩余天数标签'] = count_by_day['剩余天数'].apply(lambda x: f"剩 {x} 天")
                
                fig_rescue = px.bar(
                    count_by_day, x='剩余天数标签', y='单量',
                    text='单量',
                    title="未来3天到期分布",
                    color='剩余天数', color_continuous_scale='Reds_r'
                )
                st.plotly_chart(fig_rescue, use_container_width=True)

            with c_list:
                st.markdown("#### 优先排产清单 (按时间紧迫度)")
                
                def highlight_urgent(val):
                    if val == 1: return 'background-color: #ffcccc; color: #cc0000; font-weight: bold'
                    if val == 2: return 'background-color: #ffe6cc; color: #cc6600'
                    return ''

                view_cols = ['时间差指标', '要求交期', '样品传递单号', '客户', '设计员']
                
                st.dataframe(
                    rescue_df.sort_values('时间差指标')[view_cols].style.map(highlight_urgent, subset=['时间差指标']),
                    column_config={
                        "时间差指标": st.column_config.NumberColumn("倒计时", format=" 剩 %d 天"),
                        "要求交期": st.column_config.DateColumn("Deadline", format="MM-DD"),
                    },
                    use_container_width=True,
                    height=400,
                    hide_index=True
                )
        else:
            st.success("未来3天内没有即将到期的订单")

else:
    st.info("请上传数据文件。")

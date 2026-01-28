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

# --- 1. CSS 美化 (保持高级感) ---
st.markdown("""
<style>
    .stApp { background-color: #f4f5f7; font-family: 'PingFang SC', sans-serif; }
    h1, h2, h3 { color: #172b4d; font-weight: 700; }
    div[data-testid="stMetric"] {
        background-color: #ffffff; padding: 15px; border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05); border-left: 5px solid #0052cc;
    }
    /* Tab 样式优化 */
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #ffffff; border-radius: 4px; font-weight: 600; }
    .stTabs [aria-selected="true"] { background-color: #deebff; color: #0052cc; }
    /* 风险提示条 */
    .risk-alert { padding: 10px; border-radius: 5px; margin-bottom: 10px; font-weight: bold;}
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
    
    # 日期处理
    date_cols = ['下单日期', '要求交期', '发货日期', '技术确认日期']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # 确定计算基准日期 (支持时光倒流)
    today = pd.to_datetime(simulation_date) if simulation_date else pd.to_datetime(datetime.date.today())

    # --- 逻辑核心 ---
    
    # 1. 计算实际完成日期
    if '发货日期' in df.columns and '技术确认日期' in df.columns:
        df['完工日期'] = df['发货日期'].fillna(df['技术确认日期'])
    else:
        df['完工日期'] = pd.NaT

    # 2. 状态判定 (全面包含当前与历史)
    def evaluate_status(row):
        deadline = row.get('要求交期')
        done_date = row.get('完工日期')
        
        if pd.isnull(deadline): return "⚪ 未知"
        
        # 判断是否在“当下”已经完成
        # 如果完成日期 > 模拟今天，则在模拟视角下视为“未完成”
        is_completed = pd.notnull(done_date) and (done_date <= today)
        
        if is_completed:
            # === 历史数据判断 ===
            if done_date > deadline:
                return "⚠️ 逾期交付 (历史)" # 重点：历史问题
            else:
                return "✅ 按时交付"
        else:
            # === 当前数据判断 ===
            days_left = (deadline - today).days
            if days_left < 0:
                return "🔴 严重逾期 (进行中)" # 重点：当下火灾
            elif days_left <= 3:
                return "🟠 紧急 (3天内)" # 重点：当下预警
            else:
                return "🔵 正常进行"

    df['业务状态'] = df.apply(evaluate_status, axis=1)
    
    # 3. 辅助计算 (剩余天数/超期天数)
    # 如果未完成：显示距离截止日还有几天（负数表示已超期）
    # 如果已完成：显示超期了几天（正数表示超期天数，0表示按时）
    def calc_days_gap(row):
        deadline = row.get('要求交期')
        done_date = row.get('完工日期')
        
        if "历史" in row['业务状态']:
            # 历史逾期：实际完成日 - 要求交期 (正数)
            return (done_date - deadline).days
        elif "进行" in row['业务状态'] or "紧急" in row['业务状态'] or "严重" in row['业务状态']:
            # 进行中：要求交期 - 今天 (负数表示逾期)
            return (deadline - today).days
        else:
            return 999 # 正常完成的放最后

    df['时间差指标'] = df.apply(calc_days_gap, axis=1)

    # 4. 填充空值
    df['寄出总数量'] = df['寄出总数量'].fillna(0)
    for c in ['客户', '款式', '业务员', '设计员']:
        if c in df.columns: df[c] = df[c].fillna("未知")

    return df, today

# --- 3. 页面渲染 ---

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/factory.png", width=60)
    st.markdown("### 中兴智能工厂")
    
    st.markdown("#### ⏱️ 模拟日期")
    real_today = datetime.date.today()
    sim_date = st.date_input("基准日期", value=real_today, help="修改此日期可以查看过去某一天的生产状况")
    
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
    
    # === KPI 核心指标 (修改版：混合视角) ===
    # 逻辑：只要是逾期（不管现在还是过去）都算异常
    
    current_overdue = len(df[df['业务状态'] == "🔴 严重逾期 (进行中)"])
    current_urgent = len(df[df['业务状态'] == "🟠 紧急 (3天内)"])
    history_bad = len(df[df['业务状态'] == "⚠️ 逾期交付 (历史)"])
    
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    with col_kpi1:
        st.metric("🚨 需立即干预 (当前)", f"{current_overdue} 单", "正在发生的延误", delta_color="inverse")
    with col_kpi2:
        st.metric("🟠 3日内临期 (当前)", f"{current_urgent} 单", "即将发生的延误", delta_color="inverse")
    with col_kpi3:
        st.metric("⚠️ 历史逾期记录", f"{history_bad} 单", "已完工但超期", delta_color="off")
    with col_kpi4:
        # 总异常率
        total_orders = len(df)
        total_issues = current_overdue + history_bad
        rate = (total_issues / total_orders * 100) if total_orders > 0 else 0
        st.metric("整体履约异常率", f"{rate:.1f}%", f"总计 {total_issues} 个问题单")

    st.divider()

    tab1, tab2, tab3 = st.tabs(["1.风险与问题管控 (Risk & Issues)", "2.技术员绩效", "3.智能洞察"])

    # === Tab 1: 风险与问题管控 (核心修改) ===
    with tab1:
        st.markdown("### 📋 问题订单追踪 (Risk & Issues Log)")
        
        # 定义什么算“问题订单”：当前逾期 + 当前紧急 + 历史逾期
        problem_mask = df['业务状态'].isin([
            "🔴 严重逾期 (进行中)", 
            "🟠 紧急 (3天内)", 
            "⚠️ 逾期交付 (历史)"
        ])
        
        problem_df = df[problem_mask].copy()
        
        # 排序逻辑：
        # 1. 优先级：当前严重 > 当前紧急 > 历史逾期
        # 2. 辅助排序：时间差
        status_priority = {
            "🔴 严重逾期 (进行中)": 1,
            "🟠 紧急 (3天内)": 2,
            "⚠️ 逾期交付 (历史)": 3
        }
        problem_df['priority'] = problem_df['业务状态'].map(status_priority)
        
        # 最终排序：先按优先级，同优先级按时间差排
        display_df = problem_df.sort_values(['priority', '时间差指标'])
        
        # === 智能提示逻辑 ===
        if current_overdue == 0 and current_urgent == 0:
            if history_bad > 0:
                st.info("✅ 当前无进行中的风险订单。👇 **为您展示历史逾期记录，供复盘分析：**")
            else:
                st.success("🎉 太棒了！当前无风险，且历史上也没有逾期记录。")
        else:
            st.warning(f"⚠️ 发现 {current_overdue + current_urgent} 个进行中的风险订单，请优先处理！(列表下部包含 {history_bad} 个历史逾期记录)")

        # 展示表格
        if not display_df.empty:
            view_cols = ['业务状态', '样品传递单号', '客户', '款式', '设计员', '要求交期', '完工日期', '时间差指标']
            
            # 颜色映射
            def highlight_row(val):
                s = str(val)
                if "严重" in s: return 'background-color: #ffe6e6; color: #b30000; font-weight: bold' # 浅红底深红字
                if "紧急" in s: return 'background-color: #fff8e1; color: #b38f00; font-weight: bold' # 浅黄底深黄字
                if "历史" in s: return 'color: #e65100; font-weight: bold' # 橙色字
                return ''

            st.dataframe(
                display_df[view_cols].style.map(highlight_row, subset=['业务状态']),
                column_config={
                    "业务状态": st.column_config.TextColumn("状态", width="medium"),
                    "时间差指标": st.column_config.NumberColumn(
                        "剩余/超期天数", 
                        format="%d 天",
                        help="对于进行中：负数代表已逾期天数；对于历史：正数代表超期了多少天"
                    ),
                    "要求交期": st.column_config.DateColumn("要求交期", format="YYYY-MM-DD"),
                    "完工日期": st.column_config.DateColumn("实际完工", format="YYYY-MM-DD"),
                },
                use_container_width=True,
                height=600
            )
        
        # 底部简单分析图
        if not display_df.empty:
            st.divider()
            c1, c2 = st.columns(2)
            with c1:
                # 哪个客户的问题单最多？
                bad_cust = display_df['客户'].value_counts().reset_index()
                bad_cust.columns = ['客户', '问题单数']
                fig = px.bar(bad_cust.head(10), x='客户', y='问题单数', title="🛑 问题订单最多的客户 (Top 10)", color='问题单数', color_continuous_scale='Reds')
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                # 哪种类型最容易出问题？
                bad_type = df[df['业务状态'].str.contains("逾期") | df['业务状态'].str.contains("紧急")]['样品类型'].value_counts().reset_index()
                bad_type.columns = ['样品类型', '异常频次']
                fig2 = px.pie(bad_type, values='异常频次', names='样品类型', title="🛑 异常订单类型分布")
                st.plotly_chart(fig2, use_container_width=True)

    # === Tab 2: 绩效 (微调逻辑，纳入历史数据) ===
    with tab2:
        st.markdown("### 🏆 技术部效能矩阵")
        st.caption("综合评估：纳入历史所有已完工数据进行分析。")
        
        perf_df = df.groupby('设计员').agg(
            总接单量=('样品传递单号', 'nunique'),
            总打样数=('寄出总数量', 'sum')
        ).reset_index()
        
        # 只看已完工的（包含历史逾期）
        finished_df = df[df['业务状态'].isin(["✅ 按时交付", "⚠️ 逾期交付 (历史)"])]
        
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
            full_stats, x="总打样数", y="及时率", size="总接单量", color="及时率",
            text="设计员", color_continuous_scale="RdYlGn", size_max=60,
            title="人员效能矩阵：工作量 vs 及时率"
        )
        # 增加基准线
        fig_bubble.add_hline(y=90, line_dash="dot", annotation_text="90% 及格线", annotation_position="bottom right")
        
        st.plotly_chart(fig_bubble, use_container_width=True)

    # === Tab 3: 智能洞察 (保持原样) ===
    with tab3:
        st.markdown("### 🧠 业务深层洞察")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 📅 月度逾期率趋势")
            # 按月份统计逾期占比
            df['月'] = df['要求交期'].dt.to_period('M').astype(str)
            trend_df = df.groupby('月').apply(lambda x: (x['业务状态'].str.contains('逾期')).sum() / len(x) * 100).reset_index(name='逾期率')
            fig_trend = px.line(trend_df, x='月', y='逾期率', markers=True, title="月度逾期率变化 (%)")
            st.plotly_chart(fig_trend, use_container_width=True)

        with c2:
            st.markdown("#### 👔 业务员与逾期关联")
            sales_delay = df[df['业务状态'].str.contains('逾期')].groupby('业务员').size().reset_index(name='逾期单数')
            sales_delay = sales_delay.sort_values('逾期单数', ascending=False).head(10)
            fig_sales = px.bar(sales_delay, x='逾期单数', y='业务员', orientation='h', title="各业务员名下逾期单数")
            st.plotly_chart(fig_sales, use_container_width=True)

        st.divider()
        st.markdown("### 🤖 风险预警AI")
        
        # 训练集：所有已完工的历史数据
        train_df = df[df['业务状态'].isin(["✅ 按时交付", "⚠️ 逾期交付 (历史)"])].copy()
        # 预测集：所有未完工的进行中数据
        pred_df = df[df['业务状态'].isin(["🔵 正常进行", "🟠 紧急 (3天内)", "🔴 严重逾期 (进行中)"])].copy()
        
        if len(train_df) > 10 and len(pred_df) > 0:
            train_df['Is_Late'] = train_df['业务状态'].apply(lambda x: 1 if "逾期" in str(x) else 0)
            
            le_cust = LabelEncoder()
            all_cust = pd.concat([train_df['客户'].astype(str), pred_df['客户'].astype(str)]).unique()
            le_cust.fit(all_cust)
            
            train_df['Cust_Code'] = le_cust.transform(train_df['客户'].astype(str))
            pred_df['Cust_Code'] = le_cust.transform(pred_df['客户'].astype(str))
            
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(train_df[['Cust_Code', '寄出总数量']], train_df['Is_Late'])
            
            probs = model.predict_proba(pred_df[['Cust_Code', '寄出总数量']])[:, 1]
            pred_df['风险指数'] = probs
            
            st.dataframe(
                pred_df.sort_values('风险指数', ascending=False)[['样品传递单号', '客户', '业务员', '风险指数']],
                column_config={"风险指数": st.column_config.ProgressColumn("预测延误率", format="%.0f%%")},
                use_container_width=True
            )
        else:
            if len(pred_df) == 0:
                st.info("当前无进行中订单，无需预测。")
            else:
                st.warning("历史数据不足，AI 暂无法启动。")

else:
    st.info("请在左侧上传数据文件。")
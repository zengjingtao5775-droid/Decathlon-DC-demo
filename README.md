# ZX R&D KPI Dashboard

基于 `DCKPI Dashboard.xlsx` 工作表1的需求制作的 Streamlit KPI 看板。后续维护时，只需更新 Excel 的工作表2并在网页上传整份文件。

Dashboard 默认连接指定的公开 Google 表格，并每 5 分钟自动同步一次；也可手动点击刷新。Excel 上传和内置模板保留为备用数据源。

## 本地运行

```bash
cd "/Users/eric/Documents/ZX DC dashboard"
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

浏览器打开 `http://localhost:8501`。

## Excel 数据要求

- 系统优先读取名为 `工作表2` 的工作表；没有该名称时读取第二张工作表。
- 前三列依次为 `Job`、`Name`、`2级KPI`。
- 第 4 列起为月份，推荐使用 Excel 日期并设置显示格式 `yyyy/m`。
- 比率支持 `95%`、`0.95` 或 `95`。
- 次数类 KPI 名称应包含“次数”“数量”或 `count`。
- 未准时原因使用独立 KPI 行“未准时提交的原因”，在对应月份填写文字说明。
- 周期类 KPI 名称应包含 `L/T`、`Lead Time`、`周期`、`时长`或`天数`。

## Google 表格连接

- Google 表格需要设置为“知道链接的任何人可查看”。
- Dashboard 通过 Google 官方 Excel 导出地址读取数据，不需要 API Key。
- 维护时只需更新 Google 表格的工作表2。
- 左侧可点击“立即刷新 Google 数据”跳过 5 分钟缓存。

## Streamlit Community Cloud 部署

1. 将本目录推送到 GitHub。
2. 登录 [Streamlit Community Cloud](https://share.streamlit.io/)。
3. 选择仓库、分支及入口文件 `app.py`。
4. 点击 Deploy。项目不需要密钥或数据库。

## 当前业务口径

- 团队/职位比率为各 KPI 的等权平均，因为原始表暂未提供分子、分母。
- 工作表1明确的目标已内置：PAP 95%、TF 95%、BOM 100%、GO PROD 100%、3D 100%。
- 其他 RFT 默认目标为 95%，其他准时率默认目标为 95% 或按明确规则设为 100%。
- 当前原始表没有 L/T 数据，补充周期类 KPI 行后系统会自动识别。

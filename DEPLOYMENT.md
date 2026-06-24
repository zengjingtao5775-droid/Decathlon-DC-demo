# Streamlit Community Cloud 部署

## 配置

- Git 分支：`main`
- 入口文件：`app.py`
- Python：`3.12`（由 `runtime.txt` 指定）
- 不需要 Streamlit Secrets

## 数据源

应用默认读取公开的 Google 表格，每 5 分钟自动刷新，并允许手动刷新。

Google 表格必须保持“知道链接的任何人可查看”，否则云端无法下载数据。

## 部署

1. 将仓库推送到 GitHub。
2. 在 Streamlit Community Cloud 创建应用。
3. 选择该仓库和 `main` 分支。
4. Main file path 填写 `app.py`。
5. 部署完成后检查 Google 表格连接状态和 KPI 页面。

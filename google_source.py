from __future__ import annotations

import re
from urllib.request import Request, urlopen


GOOGLE_SHEET_PATTERN = re.compile(r"/spreadsheets/d/([a-zA-Z0-9_-]+)")


def extract_google_sheet_id(url: str) -> str:
    match = GOOGLE_SHEET_PATTERN.search(url.strip())
    if not match:
        raise ValueError("无法从链接中识别 Google 表格 ID。")
    return match.group(1)


def google_sheet_export_url(url: str) -> str:
    spreadsheet_id = extract_google_sheet_id(url)
    return (
        f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}"
        "/export?format=xlsx"
    )


def download_google_sheet(url: str, timeout: int = 30) -> bytes:
    request = Request(
        google_sheet_export_url(url),
        headers={"User-Agent": "Mozilla/5.0 ZX-RD-KPI-Dashboard/1.0"},
    )
    with urlopen(request, timeout=timeout) as response:
        content = response.read()
    if not content.startswith(b"PK"):
        raise ValueError(
            "Google 表格没有返回有效的 Excel 文件，请确认共享权限为“知道链接的任何人可查看”。"
        )
    return content


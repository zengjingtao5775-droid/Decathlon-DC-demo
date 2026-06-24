from __future__ import annotations

from numbers import Number
from pathlib import Path
from typing import BinaryIO

import pandas as pd


REQUIRED_COLUMNS = ("Job", "Name", "KPI")


def _text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def parse_month_header(value: object) -> pd.Timestamp | None:
    """Convert Excel date headers, datetime headers, or strings such as 2026/2."""
    if isinstance(value, (pd.Timestamp,)):
        return value.to_period("M").to_timestamp()

    if hasattr(value, "year") and hasattr(value, "month"):
        try:
            return pd.Timestamp(value).to_period("M").to_timestamp()
        except (TypeError, ValueError):
            pass

    if isinstance(value, Number) and not isinstance(value, bool):
        try:
            parsed = pd.to_datetime(
                float(value), unit="D", origin="1899-12-30", errors="raise"
            )
            return parsed.to_period("M").to_timestamp()
        except (TypeError, ValueError, OverflowError):
            return None

    text = _text(value)
    if not text:
        return None
    try:
        parsed = pd.to_datetime(text, errors="raise")
        return parsed.to_period("M").to_timestamp()
    except (TypeError, ValueError):
        return None


def classify_kpi(kpi: str) -> tuple[str, str]:
    key = kpi.lower().replace(" ", "")
    if "原因" in key or "reason" in key:
        return "reason", "未准时原因"
    if any(token in key for token in ("未准时", "次数", "count", "数量")):
        return "count", "异常次数"
    if any(token in key for token in ("l/t", "leadtime", "周期", "时长", "天数")):
        return "duration", "L/T"
    if any(token in key for token in ("一次通过", "rft", "bom", "pap", "tf", "3d")):
        return "rate", "RFT"
    if any(token in key for token in ("ontime", "准时", "准确率", "交付")):
        return "rate", "准时交付"
    return "rate", "其他比率"


def target_for_kpi(kpi: str, metric_type: str) -> float | None:
    key = kpi.lower().replace(" ", "")
    if metric_type == "reason":
        return None
    if metric_type == "count":
        return 0.0
    if metric_type == "duration":
        return None
    if any(token in key for token in ("bom", "3d", "goprod", "开发准时", "sotpace")):
        return 1.0
    if any(token in key for token in ("pap", "tf", "样品一次", "rft")):
        return 0.95
    return 0.95


def load_kpi_data(
    source: str | Path | BinaryIO,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Read the second worksheet and reshape the wide monthly table to long form."""
    excel = pd.ExcelFile(source, engine="openpyxl")
    if "工作表2" in excel.sheet_names:
        sheet_name = "工作表2"
    elif len(excel.sheet_names) >= 2:
        sheet_name = excel.sheet_names[1]
    else:
        sheet_name = excel.sheet_names[0]

    raw = pd.read_excel(excel, sheet_name=sheet_name)
    if raw.shape[1] < 4:
        raise ValueError("数据表至少需要 Job、Name、2级KPI 和一个月份列。")

    first_three = list(raw.columns[:3])
    raw = raw.rename(
        columns={
            first_three[0]: "Job",
            first_three[1]: "Name",
            first_three[2]: "KPI",
        }
    )

    raw["Job"] = raw["Job"].map(_text)
    raw["Name"] = raw["Name"].map(_text)
    raw["KPI"] = raw["KPI"].map(_text)
    raw = raw[(raw["Job"] != "") & (raw["Name"] != "") & (raw["KPI"] != "")].copy()

    month_columns: dict[object, pd.Timestamp] = {}
    for column in raw.columns[3:]:
        parsed = parse_month_header(column)
        if parsed is not None:
            month_columns[column] = parsed
    if not month_columns:
        raise ValueError("未识别到月份列。月份表头请使用 Excel 日期或 YYYY/M 格式。")

    melted = raw.melt(
        id_vars=list(REQUIRED_COLUMNS),
        value_vars=list(month_columns),
        var_name="MonthColumn",
        value_name="RawValue",
    )
    melted["Month"] = melted["MonthColumn"].map(month_columns)
    melted["Value"] = pd.to_numeric(melted["RawValue"], errors="coerce")

    classifications = melted["KPI"].map(classify_kpi)
    melted["MetricType"] = classifications.map(lambda item: item[0])
    melted["KPIGroup"] = classifications.map(lambda item: item[1])
    reason_mask = melted["MetricType"].eq("reason")
    melted["Reason"] = melted["RawValue"].map(_text).where(reason_mask, "")

    invalid_numeric = int(
        (
            ~reason_mask
            & melted["RawValue"].notna()
            & melted["Value"].isna()
        ).sum()
    )
    valid_reason = reason_mask & melted["Reason"].ne("")
    melted = melted[melted["Value"].notna() | valid_reason].copy()

    rate_mask = melted["MetricType"].eq("rate")
    percent_mask = rate_mask & melted["Value"].abs().gt(1.5)
    melted.loc[percent_mask, "Value"] = melted.loc[percent_mask, "Value"] / 100

    melted["Target"] = [
        target_for_kpi(kpi, metric_type)
        for kpi, metric_type in zip(melted["KPI"], melted["MetricType"])
    ]
    melted["TargetMet"] = melted.apply(
        lambda row: (
            row["Value"] <= row["Target"]
            if row["MetricType"] == "count" and pd.notna(row["Target"])
            else row["Value"] >= row["Target"]
            if pd.notna(row["Target"])
            else pd.NA
        ),
        axis=1,
    )

    duplicate_count = int(
        melted.duplicated(["Job", "Name", "KPI", "Month"], keep=False).sum()
    )
    melted = (
        melted.sort_values(["Month", "Job", "Name", "KPI"])
        .reset_index(drop=True)
        .drop(columns=["MonthColumn", "RawValue"])
    )

    info = {
        "sheet_name": sheet_name,
        "sheet_names": excel.sheet_names,
        "row_count": len(raw),
        "data_points": len(melted),
        "invalid_numeric": invalid_numeric,
        "duplicate_count": duplicate_count,
        "month_count": melted["Month"].nunique(),
        "employee_count": melted["Name"].nunique(),
        "job_count": melted["Job"].nunique(),
        "reason_count": int(melted["MetricType"].eq("reason").sum()),
    }
    return melted, info


def period_detail(data: pd.DataFrame) -> pd.DataFrame:
    data = data[data["MetricType"].ne("reason")].copy()
    if data.empty:
        return pd.DataFrame()

    latest_month = data["Month"].max()
    latest = (
        data[data["Month"].eq(latest_month)]
        .groupby(["Job", "Name", "KPI", "MetricType", "KPIGroup"], as_index=False)
        .agg(Latest=("Value", "mean"), Target=("Target", "mean"))
    )
    average = (
        data.groupby(["Job", "Name", "KPI"], as_index=False)
        .agg(PeriodAverage=("Value", "mean"), Months=("Month", "nunique"))
    )
    result = latest.merge(average, on=["Job", "Name", "KPI"], how="outer")

    def status(row: pd.Series) -> str:
        if pd.isna(row.get("Latest")) or pd.isna(row.get("Target")):
            return "—"
        if row["MetricType"] == "count":
            return "达标" if row["Latest"] <= row["Target"] else "需关注"
        return "达标" if row["Latest"] >= row["Target"] else "需关注"

    result["Status"] = result.apply(status, axis=1)
    return result.sort_values(["Job", "Name", "KPI"]).reset_index(drop=True)

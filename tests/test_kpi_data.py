import unittest
from pathlib import Path

import pandas as pd

from google_source import extract_google_sheet_id, google_sheet_export_url
from kpi_data import classify_kpi, load_kpi_data, parse_month_header


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "DCKPI Dashboard.xlsx"


class KPIDataTests(unittest.TestCase):
    def test_google_sheet_url(self):
        url = (
            "https://docs.google.com/spreadsheets/d/"
            "1msT7WcjkYva2yojVNrET2xJfqRS00xH85j1D85Ne36Q/edit?usp=sharing"
        )
        self.assertEqual(
            extract_google_sheet_id(url),
            "1msT7WcjkYva2yojVNrET2xJfqRS00xH85j1D85Ne36Q",
        )
        self.assertTrue(google_sheet_export_url(url).endswith("/export?format=xlsx"))

    def test_excel_serial_header(self):
        self.assertEqual(parse_month_header(46071), pd.Timestamp("2026-02-01"))

    def test_kpi_classification(self):
        self.assertEqual(classify_kpi("PAP 一次通过率"), ("rate", "RFT"))
        self.assertEqual(classify_kpi("未准时提交的次数"), ("count", "异常次数"))
        self.assertEqual(
            classify_kpi("未准时提交的原因"), ("reason", "未准时原因")
        )

    def test_sample_workbook(self):
        data, info = load_kpi_data(SAMPLE)
        self.assertEqual(info["sheet_name"], "工作表2")
        self.assertEqual(info["job_count"], 4)
        self.assertEqual(info["employee_count"], 5)
        self.assertEqual(info["month_count"], 5)
        self.assertEqual(len(data), 74)
        self.assertEqual(info["reason_count"], 4)
        self.assertAlmostEqual(
            data.loc[data["KPI"].eq("样品一次通过率"), "Value"].min(),
            0.7143,
        )
        reasons = data[data["MetricType"].eq("reason")]
        self.assertIn("软件研究", reasons["Reason"].tolist())
        self.assertIn("样品间排单拥挤", reasons["Reason"].tolist())
        self.assertTrue(reasons["Target"].isna().all())


if __name__ == "__main__":
    unittest.main()

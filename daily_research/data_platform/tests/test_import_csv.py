import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from daily_research.data_lake import ResearchDataLake, load_policy_inputs_from_lake
from daily_research.data_platform.contracts import DataDomain
from daily_research.data_platform.import_csv import CsvImportConfig, run_import_csv


class DataPlatformCsvImportTest(unittest.TestCase):
    def test_long_format_csv_imports_to_bronze_silver_and_lake(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            csv_path = root / "market.csv"
            pd.DataFrame(
                [
                    {"symbol": "000001.SZ", "trade_date": "2026-01-05", "open": 10, "high": 11, "low": 9, "close": 10.5, "volume": 100, "amount": 1050},
                    {"symbol": "000300.SH", "trade_date": "2026-01-05", "open": 4000, "high": 4010, "low": 3990, "close": 4005, "volume": 300, "amount": 1201500},
                ]
            ).to_csv(csv_path, index=False)

            result = run_import_csv(
                CsvImportConfig(
                    input_path=csv_path,
                    lake_root=root / "lake",
                    domain=DataDomain.MARKET_DAILY,
                    as_of_date="2026-01-05",
                    source_name="manual_csv",
                    benchmark="000300.SH",
                )
            )
            manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
            bronze_exists = Path(result.bronze_path).exists()
            silver_exists = Path(result.silver_path).exists()
            prepared = load_policy_inputs_from_lake(
                lake=ResearchDataLake(root / "lake"),
                dataset_id=result.registered_dataset_id,
                start_date="2026-01-05",
                end_date="2026-01-05",
                benchmark="000300.SH",
                min_trading_days=1,
            )

        self.assertEqual(result.status, "ok")
        self.assertEqual(manifest["provider_plan"], "csv_import")
        self.assertTrue(bronze_exists)
        self.assertTrue(silver_exists)
        self.assertEqual(prepared.universe, ("000001.SZ",))

    def test_folder_per_symbol_csv_imports_to_market_bundle(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            folder = root / "folder"
            folder.mkdir()
            pd.DataFrame(
                [{"trade_date": "2026-01-05", "open": 10, "high": 11, "low": 9, "close": 10.5, "volume": 100, "amount": 1050}]
            ).to_csv(folder / "000001.SZ.csv", index=False)
            pd.DataFrame(
                [{"trade_date": "2026-01-05", "open": 4000, "high": 4010, "low": 3990, "close": 4005, "volume": 300, "amount": 1201500}]
            ).to_csv(folder / "000300.SH.csv", index=False)

            result = run_import_csv(
                CsvImportConfig(
                    input_path=folder,
                    lake_root=root / "lake",
                    domain=DataDomain.MARKET_DAILY,
                    as_of_date="2026-01-05",
                    source_name="vendor_patch",
                    benchmark="000300.SH",
                )
            )

        self.assertEqual(result.status, "ok")
        self.assertTrue(result.registered_dataset_id.startswith("policy_input_bundle__"))


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest import mock
from pathlib import Path
import subprocess
import sys

from daily_research.baseline import data_provider
from daily_research.baseline.data_provider import get_latest_completed_trading_date
from daily_research.continuous_policy.state_builder import prepare_policy_inputs
from quant_data_platform.lake import build_research_database
from quant_data_platform.domains.contracts import validate_provider_name


class TdxFreeGuardTest(unittest.TestCase):
    def test_formal_prepare_policy_inputs_rejects_tdx_family_sources(self) -> None:
        for data_source in ["tq", "tdx", "pytdx", "mootdx"]:
            with self.subTest(data_source=data_source):
                with self.assertRaisesRegex(ValueError, "TDX-family.*refresh_daily"):
                    prepare_policy_inputs(
                        pool_name="learned_all_a",
                        start_date="2026-01-05",
                        end_date="2026-01-06",
                        data_source=data_source,
                    )

    def test_build_research_database_parser_no_longer_accepts_tq_source(self) -> None:
        parser = build_research_database.build_parser()

        with self.assertRaises(SystemExit):
            parser.parse_args(["--data-source", "tq"])

        args = parser.parse_args(["--data-source", "lake"])
        self.assertEqual(args.data_source, "lake")

    def test_provider_guard_rejects_tdx_family_aliases(self) -> None:
        for provider in ["tqcenter", "tq", "tdx", "pytdx", "mootdx"]:
            with self.subTest(provider=provider):
                with self.assertRaisesRegex(ValueError, "TDX-family"):
                    validate_provider_name(provider)

    def test_latest_completed_date_no_longer_imports_tq(self) -> None:
        with mock.patch(
            "daily_research.baseline.data_provider._try_import_tq",
            side_effect=AssertionError("latest date must not import TDX"),
        ):
            value = get_latest_completed_trading_date(reference_ts="2026-01-07 16:00")

        self.assertEqual(value, "2026-01-07")

    def test_latest_completed_date_does_not_roll_weekend_to_future(self) -> None:
        self.assertEqual(
            get_latest_completed_trading_date(reference_ts="2026-05-23 20:00"),
            "2026-05-22",
        )
        self.assertEqual(
            get_latest_completed_trading_date(reference_ts="2026-05-24 20:00"),
            "2026-05-22",
        )

    def test_internal_refresh_daily_help_bootstraps_project_imports(self) -> None:
        repo_root = Path(__file__).resolve().parents[3]

        result = subprocess.run(
            [sys.executable, "-m", "quant_data_platform.ingest.refresh_daily", "--help"],
            cwd=str(repo_root),
            text=True,
            capture_output=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--as-of-date", result.stdout)

    def test_legacy_tq_import_is_disabled_without_explicit_environment_flag(self) -> None:
        with mock.patch.dict("os.environ", {"DAILY_RESEARCH_ALLOW_TDX_FAMILY": ""}, clear=False):
            self.assertIsNone(data_provider._try_import_tq())

    def test_formal_data_platform_and_policy_entrypoints_do_not_import_tqcenter(self) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        formal_paths = [
            repo_root / "quant_data_platform" / "src" / "quant_data_platform" / "domains" / "contracts.py",
            repo_root / "quant_data_platform" / "src" / "quant_data_platform" / "provider_manager.py",
            repo_root / "quant_data_platform" / "src" / "quant_data_platform" / "providers.py",
            repo_root / "quant_data_platform" / "src" / "quant_data_platform" / "ingest" / "refresh_daily.py",
            repo_root / "quant_data_platform" / "src" / "quant_data_platform" / "ingest" / "import_csv.py",
            repo_root / "daily_research" / "path_policy" / "run_alpha_path20_protocol.py",
            repo_root / "daily_research" / "continuous_policy" / "evaluate_policy.py",
            repo_root / "daily_research" / "continuous_policy" / "train_policy.py",
            repo_root / "daily_research" / "continuous_policy" / "run_continuous_policy_protocol.py",
        ]
        offenders = []
        for path in formal_paths:
            files = path.rglob("*.py") if path.is_dir() else [path]
            for file_path in files:
                if "tests" in file_path.parts:
                    continue
                text = file_path.read_text(encoding="utf-8")
                if "from tqcenter import" in text or "import tqcenter" in text:
                    offenders.append(str(file_path.relative_to(repo_root)))

        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()

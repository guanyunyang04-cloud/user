from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

from quant_data_platform.core.json_io import json_safe
from quant_data_platform.core.paths import qdp_paths
from quant_data_platform.core.registry import migrate_legacy_registry, registry_status
from quant_data_platform.event_packs.traditional_alpha import EventPackConfig, build_traditional_event_alpha_pack
from quant_data_platform.lake.adapters import audit_inventory, build_bundle, bundle_summary_dict, cleanup_dry_run
from quant_data_platform.memmap.incremental import (
    IncrementalPlanConfig,
    compose_sharded_memmap,
    freeze_sharded_memmap,
    plan_incremental_memmap,
)
from quant_data_platform.memmap.coverage_audit import audit_sharded_memmap_feature_coverage
from quant_data_platform.memmap.sharded import ShardedMemmapConfig, build_sharded_memmap, write_sharded_memmap_plan
from quant_data_platform.memmap.training_pack import (
    RegimeTrainingPackConfig,
    TrainingPackConfig,
    build_regime_training_pack,
    build_training_pack,
)
from quant_data_platform.memmap.validation import validate_active_memmap


PASSTHROUGH_COMMAND_MODULES: dict[str, str] = {
    "provider-health": "quant_data_platform.provider_health",
    "refresh-daily": "quant_data_platform.ingest.refresh_daily",
    "daily-update": "quant_data_platform.ingest.daily_update",
    "import-csv": "quant_data_platform.ingest.import_csv",
    "baostock-backfill": "quant_data_platform.ingest.baostock_backfill",
    "build-intraday-daily-features": "quant_data_platform.ingest.build_intraday_daily_features",
    "combine-domain-datasets": "quant_data_platform.ingest.combine_domain_datasets",
    "combine-sharded-domain-datasets": "quant_data_platform.ingest.combine_sharded_domain_datasets",
    "import-external-quant-zip": "quant_data_platform.ingest.import_external_quant_zip",
    "recover-external-quant-zip-import": "quant_data_platform.ingest.recover_external_quant_zip_import",
    "import-tdx-5m": "quant_data_platform.ingest.import_tdx_5m",
    "import-tdx-daily": "quant_data_platform.ingest.import_tdx_daily",
    "audit-gold-dataset": "quant_data_platform.lake.audit_gold_dataset",
    "build-canonical-policy-bundle": "quant_data_platform.lake.build_canonical_policy_bundle",
    "build-gold-training-dataset": "quant_data_platform.lake.build_gold_training_dataset",
    "build-pool-view": "quant_data_platform.lake.build_pool_view",
    "build-research-database": "quant_data_platform.lake.build_research_database",
    "build-sector-board-view": "quant_data_platform.lake.build_sector_board_view",
    "build-v2-status-sidecar": "quant_data_platform.lake.build_v2_status_sidecar",
    "canonical-audit": "quant_data_platform.lake.canonical_audit",
    "import-legacy-training-caches": "quant_data_platform.lake.import_legacy_training_caches",
    "import-traditional-baostock-v2-snapshot": "quant_data_platform.lake.import_traditional_baostock_v2_snapshot",
    "import-traditional-pit-status-sidecar": "quant_data_platform.lake.import_traditional_pit_status_sidecar",
    "policy-input-audit": "quant_data_platform.lake.policy_input_audit",
    "v2-dataset-contract-audit": "quant_data_platform.lake.v2_dataset_contract_audit",
}


def _print(payload: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2))
        return
    for key, value in payload.items():
        if isinstance(value, (dict, list)):
            print(f"{key}: {json.dumps(json_safe(value), ensure_ascii=False)}")
        else:
            print(f"{key}: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qdp", description="Quant Data Platform governance CLI.")
    parser.add_argument("--workspace-root", default="", help="Workspace root, defaults to auto-detection.")
    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status", help="Show canonical registry, bundle, and active memmap status.")
    status.add_argument("--json", action="store_true")

    init_registry = sub.add_parser("init-registry", help="Migrate canonical_data registry into quant_data_platform/registry.")
    init_registry.add_argument("--dry-run", action="store_true")
    init_registry.add_argument("--json", action="store_true")

    audit = sub.add_parser("audit", help="Write or print coverage, duplicate, orphan, and cleanup dry-run inventory.")
    audit.add_argument("--no-write", action="store_true")
    audit.add_argument("--json", action="store_true")

    bundle = sub.add_parser("build-bundle", help="Build canonical policy bundle with all v1 sidecars.")
    bundle.add_argument("--dry-run", action="store_true")
    bundle.add_argument("--json", action="store_true")

    validate = sub.add_parser("validate-memmap", help="Validate active or explicit forecast memmap manifest.")
    validate.add_argument("--manifest", default="")
    validate.add_argument("--min-universe-size", type=int, default=0)
    validate.add_argument("--min-train-rows", type=int, default=0)
    validate.add_argument("--json", action="store_true")

    feature_coverage = sub.add_parser(
        "audit-sharded-feature-coverage",
        help="Audit finite feature coverage in a QDP sharded memmap without changing registries.",
    )
    feature_coverage.add_argument("--manifest-json", required=True)
    feature_coverage.add_argument("--output-root", required=True)
    feature_coverage.add_argument("--run-tag", default="qdp_sharded_feature_coverage_audit")
    feature_coverage.add_argument("--years", default="")
    feature_coverage.add_argument("--max-shards", type=int, default=0)
    feature_coverage.add_argument("--chunk-features", type=int, default=32)
    feature_coverage.add_argument("--train-year-start", type=int, default=2012)
    feature_coverage.add_argument("--train-year-end", type=int, default=2023)
    feature_coverage.add_argument("--recent-year-start", type=int, default=2024)
    feature_coverage.add_argument("--recent-year-end", type=int, default=2025)
    feature_coverage.add_argument("--low-rate-threshold", type=float, default=0.50)
    feature_coverage.add_argument("--train-warn-threshold", type=float, default=0.75)
    feature_coverage.add_argument("--recent-warn-threshold", type=float, default=0.90)
    feature_coverage.add_argument("--json", action="store_true")

    sharded = sub.add_parser("build-sharded-memmap", help="Build sharded canonical feature/label stores.")
    sharded.add_argument("--canonical-dataset-id", default="")
    sharded.add_argument("--profile", default="short_horizon_core_v1")
    sharded.add_argument("--start-year", type=int, default=0)
    sharded.add_argument("--end-year", type=int, default=0)
    sharded.add_argument("--symbol-block-size", type=int, default=300)
    sharded.add_argument("--max-universe-size", type=int, default=0)
    sharded.add_argument("--max-shards", type=int, default=0)
    sharded.add_argument("--lookback-days", type=int, default=60)
    sharded.add_argument("--horizon", type=int, default=20)
    sharded.add_argument("--cumulative-horizons", default="1,3,5,10,20")
    sharded.add_argument("--execution-mode", default="next_open")
    sharded.add_argument("--max-feature-columns", type=int, default=256)
    sharded.add_argument("--min-lookback-valid-ratio", type=float, default=0.80)
    sharded.add_argument("--tag", default="")
    sharded.add_argument("--workers", type=int, default=1)
    sharded.add_argument("--year-input-cache", action="store_true")
    sharded.add_argument("--force-years", default="")
    sharded.add_argument("--pool-view-id", default="")
    sharded.add_argument("--sector-board-view-id", default="")
    sharded.add_argument("--include-static-context", action="store_true")
    sharded.add_argument("--static-context-fields", default="symbol,exchange,industry")
    sharded.add_argument("--no-resume", action="store_true")
    sharded.add_argument("--dry-run", action="store_true")
    sharded.add_argument("--json", action="store_true")

    freeze = sub.add_parser("freeze-sharded-memmap", help="Mark a completed sharded memmap as a reusable frozen base.")
    freeze.add_argument("--manifest", default="")
    freeze.add_argument("--tag", default="")
    freeze.add_argument("--dry-run", action="store_true")
    freeze.add_argument("--json", action="store_true")

    incremental = sub.add_parser("plan-incremental-memmap", help="Plan tail-year shard rebuilds from a frozen sharded base.")
    incremental.add_argument("--base-manifest", default="")
    incremental.add_argument("--rebuild-start-year", type=int, default=0)
    incremental.add_argument("--rebuild-end-year", type=int, default=0)
    incremental.add_argument("--recent-years", type=int, default=1)
    incremental.add_argument("--tag", default="")
    incremental.add_argument("--no-write", action="store_true")
    incremental.add_argument("--json", action="store_true")

    compose = sub.add_parser("compose-sharded-memmap", help="Compose a frozen base and incremental shard manifests into one logical active view.")
    compose.add_argument("--base-manifest", default="")
    compose.add_argument("--overlay-manifest", action="append", default=[])
    compose.add_argument("--tag", default="")
    compose.add_argument("--activate", action="store_true")
    compose.add_argument("--dry-run", action="store_true")
    compose.add_argument("--json", action="store_true")

    training_pack = sub.add_parser("build-training-pack", help="Build a training-optimized pack from a QDP sharded memmap.")
    training_pack.add_argument("--source-manifest", default="")
    training_pack.add_argument("--output-root", default="")
    training_pack.add_argument("--tag", default="")
    training_pack.add_argument("--train-start-year", type=int, default=2012)
    training_pack.add_argument("--train-end-year", type=int, default=2023)
    training_pack.add_argument("--validation-year", type=int, default=2024)
    training_pack.add_argument("--test-year", type=int, default=2025)
    training_pack.add_argument("--max-samples-per-role", type=int, default=0)
    training_pack.add_argument("--max-samples-per-date-per-role", type=int, default=0)
    training_pack.add_argument("--feature-dtype", default="float16", choices=("float16", "float32"))
    training_pack.add_argument("--stock-chunk-size", type=int, default=64)
    training_pack.add_argument("--no-resume", action="store_true")
    training_pack.add_argument("--json", action="store_true")

    regime_pack = sub.add_parser(
        "build-regime-training-pack",
        help="Add a date-major cross-sectional auxiliary layout to a QDP training pack.",
    )
    regime_pack.add_argument("--source-training-pack", default="")
    regime_pack.add_argument("--feature-dtype", default="")
    regime_pack.add_argument("--stock-chunk-size", type=int, default=64)
    regime_pack.add_argument("--no-resume", action="store_true")
    regime_pack.add_argument("--json", action="store_true")

    event_pack = sub.add_parser("build-event-pack", help="Build a candidate event-level research dataset managed by QDP.")
    event_pack.add_argument("--source-training-pack", default="")
    event_pack.add_argument("--pit-root", default="")
    event_pack.add_argument("--output-root", default="")
    event_pack.add_argument("--tag", default="traditional_event_alpha_v1_candidate")
    event_pack.add_argument("--start-year", type=int, default=2024)
    event_pack.add_argument("--end-year", type=int, default=2024)
    event_pack.add_argument("--max-events-per-year", type=int, default=512)
    event_pack.add_argument("--warmup-years", type=int, default=1)
    event_pack.add_argument("--sell-windows", default="1,3,5,10,20")
    event_pack.add_argument("--min-signal-amount", type=float, default=1.0e8)
    event_pack.add_argument("--fee-bps", type=float, default=30.0)
    event_pack.add_argument("--slippage-bps", type=float, default=0.0)
    event_pack.add_argument("--big-loss-threshold-pct", type=float, default=-5.0)
    event_pack.add_argument("--feature-chunk-rows", type=int, default=4096)
    event_pack.add_argument("--write-event-cache", action="store_true")
    event_pack.add_argument("--no-resume", action="store_true")
    event_pack.add_argument("--json", action="store_true")

    provider_eval = sub.add_parser("provider-eval", help="Run read-only external provider quality probes managed by QDP.")
    provider_eval.add_argument("--providers", default="akshare,baostock,efinance,mootdx,cninfo,current_qdp")
    provider_eval.add_argument("--symbols", default="000001.SZ,600000.SH,300750.SZ,688001.SH,000300.SH")
    provider_eval.add_argument(
        "--windows",
        default=(
            "2010-01-04:2010-01-15,"
            "2015-06-01:2015-06-12,"
            "2020-03-02:2020-03-13,"
            "2024-06-03:2024-06-14,"
            "2025-12-01:2025-12-12,"
            "2026-06-01:2026-06-10"
        ),
    )
    provider_eval.add_argument("--run-tag", default="")
    provider_eval.add_argument("--output-root", default="")
    provider_eval.add_argument("--python-executable", default=sys.executable)
    provider_eval.add_argument("--no-cache", action="store_true")
    provider_eval.add_argument("--no-install-missing", action="store_true")
    provider_eval.add_argument("--json", action="store_true")

    for command, module_name in PASSTHROUGH_COMMAND_MODULES.items():
        parser_for_command = sub.add_parser(command, help=f"Run {module_name}.")
        parser_for_command.add_argument("passthrough_args", nargs=argparse.REMAINDER)

    cleanup = sub.add_parser("cleanup", help="Generate cleanup dry-run plan. This command never deletes files in v1.")
    cleanup.add_argument("--dry-run", action="store_true", default=True)
    cleanup.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    for index, item in enumerate(raw_argv):
        if item in PASSTHROUGH_COMMAND_MODULES:
            return _run_passthrough_command(item, raw_argv[index + 1 :])
    args = build_parser().parse_args(raw_argv)
    if args.command in PASSTHROUGH_COMMAND_MODULES:
        return _run_passthrough_command(args.command, list(getattr(args, "passthrough_args", []) or []))
    paths = qdp_paths(args.workspace_root or None)
    if args.command == "status":
        _print(registry_status(paths), as_json=bool(args.json))
        return 0
    if args.command == "init-registry":
        _print(migrate_legacy_registry(paths, write=not bool(args.dry_run)), as_json=bool(args.json))
        return 0
    if args.command == "audit":
        inventory = audit_inventory(paths, write=not bool(args.no_write))
        summary = {
            "status": inventory.get("status", ""),
            "dataset_count": inventory.get("dataset_count", 0),
            "duplicate_group_count": len(inventory.get("duplicate_dataset_groups", []) or []),
            "orphan_fingerprint_dir_count": len(inventory.get("filesystem", {}).get("orphan_fingerprint_dirs", []) or []),
            "forecast_dat_count": len(inventory.get("memmaps", {}).get("dat_files", []) or []),
            "report_path": inventory.get("report_path", ""),
            "cleanup_dry_run": inventory.get("cleanup_dry_run", {}),
        }
        _print(summary if not bool(args.json) else inventory, as_json=bool(args.json))
        return 0
    if args.command == "build-bundle":
        summary = build_bundle(paths, write=not bool(args.dry_run))
        _print(bundle_summary_dict(summary), as_json=bool(args.json))
        return 0
    if args.command == "validate-memmap":
        report = validate_active_memmap(
            paths,
            manifest=Path(args.manifest) if str(args.manifest or "").strip() else None,
            min_universe_size=int(args.min_universe_size),
            min_train_rows=int(args.min_train_rows),
        )
        _print(report, as_json=bool(args.json))
        return 0 if str(report.get("status", "")) == "ok" else 1
    if args.command == "audit-sharded-feature-coverage":
        report = audit_sharded_memmap_feature_coverage(
            manifest_json=Path(args.manifest_json),
            output_root=Path(args.output_root),
            run_tag=str(args.run_tag or ""),
            years=str(args.years or "") or None,
            max_shards=int(args.max_shards),
            chunk_features=int(args.chunk_features),
            train_year_start=int(args.train_year_start),
            train_year_end=int(args.train_year_end),
            recent_year_start=int(args.recent_year_start),
            recent_year_end=int(args.recent_year_end),
            low_rate_threshold=float(args.low_rate_threshold),
            train_warn_threshold=float(args.train_warn_threshold),
            recent_warn_threshold=float(args.recent_warn_threshold),
        )
        _print(report, as_json=bool(args.json))
        return 0
    if args.command == "build-sharded-memmap":
        if bool(args.dry_run):
            payload = write_sharded_memmap_plan(
                paths,
                canonical_dataset_id=str(args.canonical_dataset_id or ""),
                profile=str(args.profile or ""),
                max_universe_size=int(args.max_universe_size),
                workers=int(args.workers),
                year_input_cache=bool(args.year_input_cache),
                pool_view_id=str(args.pool_view_id or ""),
                sector_board_view_id=str(args.sector_board_view_id or ""),
                include_static_context=bool(args.include_static_context),
                static_context_fields=str(args.static_context_fields or ""),
                write=True,
            )
        else:
            payload = build_sharded_memmap(
                paths,
                config=ShardedMemmapConfig(
                    canonical_dataset_id=str(args.canonical_dataset_id or ""),
                    profile=str(args.profile or ""),
                    start_year=int(args.start_year),
                    end_year=int(args.end_year),
                    symbol_block_size=int(args.symbol_block_size),
                    max_universe_size=int(args.max_universe_size),
                    max_shards=int(args.max_shards),
                    lookback_days=int(args.lookback_days),
                    horizon=int(args.horizon),
                    cumulative_horizons=str(args.cumulative_horizons or ""),
                    execution_mode=str(args.execution_mode or ""),
                    max_feature_columns=int(args.max_feature_columns),
                    min_lookback_valid_ratio=float(args.min_lookback_valid_ratio),
                    tag=str(args.tag or ""),
                    resume=not bool(args.no_resume),
                    workers=int(args.workers),
                    year_input_cache=bool(args.year_input_cache),
                    force_years=str(args.force_years or ""),
                    pool_view_id=str(args.pool_view_id or ""),
                    sector_board_view_id=str(args.sector_board_view_id or ""),
                    include_static_context=bool(args.include_static_context),
                    static_context_fields=str(args.static_context_fields or ""),
                ),
            )
        _print(payload, as_json=bool(args.json))
        return 0
    if args.command == "freeze-sharded-memmap":
        payload = freeze_sharded_memmap(
            paths,
            manifest=Path(args.manifest) if str(args.manifest or "").strip() else None,
            tag=str(args.tag or ""),
            write=not bool(args.dry_run),
        )
        _print(payload, as_json=bool(args.json))
        return 0
    if args.command == "plan-incremental-memmap":
        payload = plan_incremental_memmap(
            paths,
            config=IncrementalPlanConfig(
                base_manifest=str(args.base_manifest or ""),
                rebuild_start_year=int(args.rebuild_start_year),
                rebuild_end_year=int(args.rebuild_end_year),
                recent_years=int(args.recent_years),
                tag=str(args.tag or ""),
            ),
            write=not bool(args.no_write),
        )
        _print(payload, as_json=bool(args.json))
        return 0
    if args.command == "compose-sharded-memmap":
        payload = compose_sharded_memmap(
            paths,
            base_manifest=Path(args.base_manifest) if str(args.base_manifest or "").strip() else None,
            overlay_manifests=[Path(item) for item in list(args.overlay_manifest or [])],
            tag=str(args.tag or ""),
            activate=bool(args.activate),
            write=not bool(args.dry_run),
        )
        _print(payload, as_json=bool(args.json))
        return 0
    if args.command == "build-training-pack":
        payload = build_training_pack(
            TrainingPackConfig(
                source_manifest=str(args.source_manifest or ""),
                output_root=str(args.output_root or ""),
                tag=str(args.tag or ""),
                train_start_year=int(args.train_start_year),
                train_end_year=int(args.train_end_year),
                validation_year=int(args.validation_year),
                test_year=int(args.test_year),
                max_samples_per_role=int(args.max_samples_per_role),
                max_samples_per_date_per_role=int(args.max_samples_per_date_per_role),
                feature_dtype=str(args.feature_dtype or "float16"),
                stock_chunk_size=int(args.stock_chunk_size),
                resume=not bool(args.no_resume),
            )
        )
        _print(payload, as_json=bool(args.json))
        return 0
    if args.command == "build-regime-training-pack":
        payload = build_regime_training_pack(
            RegimeTrainingPackConfig(
                source_training_pack=str(args.source_training_pack or ""),
                feature_dtype=str(args.feature_dtype or ""),
                stock_chunk_size=int(args.stock_chunk_size),
                resume=not bool(args.no_resume),
            )
        )
        _print(payload, as_json=bool(args.json))
        return 0
    if args.command == "build-event-pack":
        payload = build_traditional_event_alpha_pack(
            EventPackConfig(
                source_training_pack=str(args.source_training_pack or ""),
                pit_root=str(args.pit_root or ""),
                output_root=str(args.output_root or ""),
                tag=str(args.tag or ""),
                start_year=int(args.start_year),
                end_year=int(args.end_year),
                max_events_per_year=int(args.max_events_per_year),
                warmup_years=int(args.warmup_years),
                sell_windows=str(args.sell_windows or ""),
                min_signal_amount=float(args.min_signal_amount),
                fee_bps=float(args.fee_bps),
                slippage_bps=float(args.slippage_bps),
                big_loss_threshold_pct=float(args.big_loss_threshold_pct),
                feature_chunk_rows=int(args.feature_chunk_rows),
                write_event_cache=bool(args.write_event_cache),
                resume=not bool(args.no_resume),
            ),
            paths=paths,
        )
        _print(payload, as_json=bool(args.json))
        return 0
    if args.command == "provider-eval":
        from quant_data_platform.provider_eval import ProviderEvalConfig, run_provider_eval

        payload = run_provider_eval(
            ProviderEvalConfig(
                providers=_split_csv(str(args.providers or "")),
                symbols=_split_csv(str(args.symbols or "")),
                windows=_parse_windows(str(args.windows or "")),
                run_tag=str(args.run_tag or ""),
                output_root=Path(args.output_root) if str(args.output_root or "").strip() else paths.data_dir / "provider_eval",
                use_cache=not bool(args.no_cache),
                install_missing=not bool(args.no_install_missing),
                python_executable=str(args.python_executable or ""),
                qdp_root_manifest=paths.root_manifest,
                canonical_manifest=paths.lake_root / "canonical" / "canonical_manifest.json",
            )
        )
        _print(payload, as_json=bool(args.json))
        return 0 if str(payload.get("status", "")) in {"ok", "empty"} else 2
    if args.command == "cleanup":
        _print(cleanup_dry_run(paths), as_json=bool(args.json))
        return 0
    raise ValueError(f"unsupported_command: {args.command}")


def _run_passthrough_command(command: str, passthrough_args: list[str]) -> int:
    module_name = PASSTHROUGH_COMMAND_MODULES[command]
    module = importlib.import_module(module_name)
    module_main = getattr(module, "main", None)
    if not callable(module_main):
        raise RuntimeError(f"{module_name} does not expose callable main(argv)")
    forwarded = list(passthrough_args or [])
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    result = module_main(forwarded)
    if isinstance(result, int):
        return result
    if isinstance(result, dict):
        print(json.dumps(json_safe(result), ensure_ascii=False, indent=2))
        return 0
    if result is None:
        return 0
    return int(result)


def _split_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(value or "").split(",") if item.strip())


def _parse_windows(value: str) -> tuple[tuple[str, str], ...]:
    windows: list[tuple[str, str]] = []
    for item in _split_csv(value):
        if ":" not in item:
            raise ValueError(f"window must be START:END, got {item}")
        start, end = item.split(":", 1)
        windows.append((start.strip(), end.strip()))
    return tuple(windows)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

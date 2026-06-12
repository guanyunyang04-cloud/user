# Path Policy Canonical Memmap Reconciliation 20260612

## Verdict
- Status: `canonical_memmap_validation / completed / research-only / shadow-only`.
- Scope: register the latest capped canonical memmap validation run so frontier freshness no longer treats it as an unreviewed output.
- Active artifact impact: `daily_research/output/active_execution_strategy.json` remains unchanged.
- Interpretation: this is dataset and memmap plumbing evidence, not model-quality evidence, not promotion evidence, and not execution unfreeze authority.

## Source Evidence
- Run tag: `canonical_short_horizon_capped100_memmap_20260611_01`.
- Study summary: `daily_research/output/path_policy/studies/canonical_short_horizon_capped100_memmap_20260611_01/study_summary.json`.
- Summary status: `completed`.
- Stage: `forecast_dataset`.
- Data source: `lake`.
- Lake dataset id: `canonical_data_v1`.
- Source policy input bundle: `policy_input_bundle__002270b729a4eabe6211ff6d`.
- Study family: `canonical_memmap_validation`.

## Registered Run Tags
- `canonical_short_horizon_capped100_memmap_20260611_01`

## Dataset And Memmap Facts
- Window: `2021-01-01` to `2025-12-31`.
- Universe: capped `100` symbols from `learned_all_a`; benchmark `000300.SH`.
- Feature profile: `raw_kline_context_v2_short_horizon_intraday_v1`.
- Feature store shape: `[1212, 100, 256]`.
- Feature count before cap: `280`; after cap: `256`.
- Intraday context feature count: `129`; adjust context feature count: `8`.
- Feature NaN ratio: `0.010332965913778878`.
- Sample count: `1536`, split as train `512`, validation `512`, test `512`.
- Sidecars used: `adjust_factor`, `intraday_daily_features`, `security_status`, `trading_calendar`, `universe_snapshot`.
- Label semantics: `next_open_entry_to_future_open`; cumulative horizons `[1, 3, 5, 10, 20]`.
- Normalization fit role: `train_only`.
- Feature leakage smoke: `passed`.

## Boundary
- This run validates a capped canonical-data memmap path and the short-horizon intraday feature profile shape.
- It does not validate full-universe scalability, full canonical sharded memmap governance, allocator behavior, live/default routing, or candidate backtest quality.
- It should not be used to claim that `canonical_data_v1` is complete; it only registers one completed capped validation artifact.

## Next Allowed Actions
- Rebuild `daily_research/brain/references/evidence_registry.json` after this reference is added.
- Run `current-frontier --json`; expected outcome is no unregistered latest run tag for `canonical_short_horizon_capped100_memmap_20260611_01`.
- Continue canonical data platform work in `quant_data_platform` before treating full shared memmap as the default research substrate.

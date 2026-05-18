# Path20 Memory-Safe Forecast Dataset Contract 2026-05-18

## Verdict
- Status: `implemented memory-safe forecast dataset/training pipe / research / shadow-only / no promotion`.
- Current Path20 research mainline remains `alpha_path20_neural_policy_v1`.
- This work repairs full-universe training feasibility; it is not a model-quality or live/default promotion claim.
- `promotion_allowed=false`, `shadow_only=true`, and `active_execution_strategy_expected_diff=none`.

## Facts
- Added `memmap` forecast dataset mode beside the existing eager mode.
- Eager mode remains the default for compatibility and small fixture tests.
- Memmap mode writes a disk feature store shaped `[date, stock, feature]`.
- Memmap mode writes a lightweight sample index with `role`, `date_pos`, `stock_pos`, `sequence_start_pos`, `label_end_pos`, `stock_seen_in_train`, `history_bucket`, and history-valid ratio fields.
- Training now consumes a dataset view abstraction and can train from eager arrays or memmap-backed lazy batches.
- Prediction CSV columns remain compatible with the Stage 1 multi-horizon contract.
- Training summaries now include `dataset_mode`, `validation_stratified_metrics`, and `test_stratified_metrics`.
- Protocol CLI accepts:
  - `--forecast-dataset-mode eager|memmap`
  - `--forecast-min-lookback-valid-ratio`
  - `--forecast-dataloader-num-workers`
  - `--forecast-prefetch-factor`
- `--max-universe-size 0` with eager forecast dataset mode is blocked with `full_universe_requires_memmap_dataset_mode`.
- Memmap mode records history-quality controls and drops samples below the configured core OHLCV lookback valid-ratio threshold.
- Feature store manifests include feature-store path/shape, feature nan ratio, and history-quality feature counts.

## Inferences
- The previous full-universe blocker was not the repaired lake itself, but the eager `[N,252,F]` training sample materialization.
- The new pipe removes the largest RAM blocker by loading lookback windows per batch.
- CPU feature-store construction remains a cost center because feature profiles still need per-date feature assembly.
- This is the required infrastructure step before full-universe Stage 1 training, but it does not yet implement true cross-stock attention.

## Boundaries
- No allocator, replay, oracle, or live/default path is changed.
- No `daily_research/output/active_execution_strategy.json` write is authorized.
- No full-universe evidence is claimed until a real repaired-bundle memmap run completes.
- Stock-to-stock relation modeling remains feature-proxy based in this version.

## Next Allowed Actions
- Run cap80 repaired-bundle memmap dataset smoke.
- Run cap80 repaired-bundle memmap training smoke.
- If both pass, plan a full-universe memmap dataset build before any full-universe training.
- After memory-safe full-universe data is stable, plan day-grouped or cross-stock attention as a separate architecture upgrade.

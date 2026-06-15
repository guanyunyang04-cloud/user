# Traditional Event Alpha V1 Candidate

## Purpose

`traditional_event_alpha_v1_candidate` is a QDP-managed event-level dataset family for `traditional_quant_research` consumption. It converts generalized strong/limit-up events into tabular rows and attaches clean QDP training-pack features.

This dataset family is candidate-stage only. It is not an active shared registry target yet.

## Current Smoke Build

- Tag: `traditional_event_alpha_v1_candidate_smoke_20260615_01`
- Dataset id: `traditional_event_alpha_v1_candidate_smoke_20260615_01__50d042d08f4455bd98c9dae6`
- Root: `quant_data_platform/data/event_packs/traditional_event_alpha_v1_candidate_smoke_20260615_01/`
- Years: `2024`
- Rows: `128`
- Partitions: `1`
- QDP feature count: `316`
- Dropped execution-state feature count: `48`
- Label count in smoke parquet: `29`
- QDP feature missing rows: `1`

Primary event type counts:

- `limit_up_core`: `57`
- `volume_atr_breakout`: `20`
- `big_up`: `17`
- `kama_breakout`: `15`
- `new_high_breakout`: `11`
- `near_limit`: `7`
- `trend_accel`: `1`

## Current Full Candidate Build

- Tag: `traditional_event_alpha_v1_candidate_2017_2026_20260615_01`
- Dataset id: `traditional_event_alpha_v1_candidate_2017_2026_20260615_01__72f307846e0a2abdf18a30cd`
- Root: `quant_data_platform/data/event_packs/traditional_event_alpha_v1_candidate_2017_2026_20260615_01/`
- Years: `2017-2026`
- Rows: `268333`
- Partitions: `10`
- Status: `candidate`
- QDP feature count: `316`
- Dropped execution-state feature count: `48`
- Label count: `29`
- QDP feature missing rows: `8172`
- QDP feature missing rate: `3.0455%`
- QDP feature nan ratio: `3.0455%`
- Split counts: `train=183233`, `validation=36854`, `test=48246`

Primary event type counts:

- `limit_up_core`: `106411`
- `big_up`: `39915`
- `near_limit`: `36031`
- `new_high_breakout`: `31154`
- `volume_atr_breakout`: `26794`
- `kama_breakout`: `26400`
- `trend_accel`: `1628`

Per-year rows:

- `2017`: `18897`
- `2018`: `16923`
- `2019`: `25723`
- `2020`: `32802`
- `2021`: `33851`
- `2022`: `32667`
- `2023`: `22370`
- `2024`: `36854`
- `2025`: `32867`
- `2026`: `15379`

Verification on 2026-06-15:

- Manifest row count, quality row count, and partition row sums all equal `268333`.
- No leaked QDP execution/portfolio state output features were found.
- All `316` QDP feature output columns and all `29` label columns exist in every yearly partition.
- The build remained candidate-stage only and did not mutate QDP active registries.
- The builder now supports incremental quality aggregation, resume, and a low-memory candidate-code prefilter. The prefilter is intended to reduce full-panel factor memory without narrowing the event definition; full-market market/industry context is recomputed separately and joined back.

## Feature Policy

Source feature pack:

- `quant_data_platform/data/memmap/training_pack/mainboard_style_structural_pack_full_2012_2025_20260613_01/qdp_training_pack_manifest.json`

The event pack reads normalized QDP feature values from the training pack memmap by `(stock, date)`. It excludes execution/portfolio state features because current traditional alpha research has no execution-state feedback loop. Excluded families include:

- `holding_*`
- `hold_days*`
- `current_weight`
- `unrealized_pnl`
- `last_action_*`
- `recent_*` trade action counters
- `portfolio_*`
- `pnl_*`
- `reentry_cooldown`
- `cash_regime_pressure`
- `exit_reentry_pressure`

## Label Policy

V1 labels are derived from the traditional generalized strong-event window labels:

- next-open entry tradeability proxy
- entry limit-up buy-block proxy
- entry-day close return after cost
- raw and cost-adjusted `D+1/D+3/D+5/D+10/D+20` close returns
- MFE/MAE by horizon
- big-loss flag by horizon

Forward tradeability path labels are not included in v1. They should be added after `style_structural_alpha_v2_label_v2` is finalized or after a dedicated execution-label builder is promoted.

## Governance

- Builder module: `quant_data_platform.event_packs.traditional_alpha`
- CLI command: `qdp build-event-pack`
- Candidate datasets live under `quant_data_platform/data/event_packs/`.
- Strategy predictions, model artifacts, top1/top2 schedules, and backtest reports remain in `traditional_quant_research`.

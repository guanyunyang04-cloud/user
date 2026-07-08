# Path Policy Seq100 Mainline Slimming Contract

日期：`2026-07-07`

## Status

`process_complexity_slimmed / concept_surface_narrowed / research_only / no_active_change`

This contract turns the current seq100 path-value research into a narrow default work surface. It does not delete historical evidence, does not change QDP active data, and does not activate execution artifacts.

## Verdict

- Status: `process_complexity_slimmed / concept_surface_narrowed / research_only / no_active_change`.
- Mainline: `seq100_todayclose_path_only/daily_only_summary_v2` is the default seq100 path-value research surface.
- Active artifact impact: unchanged. This does not touch `daily_research/output/active_execution_strategy.json`, live/default, paper/live, broker, trade plans, QDP active dataset pointers, or QDP provider state.
- Concept decision: `alpha_v2`, `path20`, `symbol_embedding`, `residual_score`, `richer_target`, `ohlcva_unified`, and `rank_heavy_top1` are no longer default concepts.

## Default Mainline

The default research line is:

```text
research_store_view
-> seq100_x32_daily_input
-> today_close_anchor
-> future60_ohlc_path
-> summary_v2_multi_horizon_ohlc
-> path_trade_value_v2
-> path_value_spread
```

Default CLI:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train --json
```

Dry-run / contract inspection:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train --dry-run --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline contract --json
```

## Fixed Profile

- Store view: `daily_research/data/research_store/views/seq100_path60_todayclose_ohlcva.json`
- Model type: `gru_path_value`
- Input: past `100 x 32` `daily_raw + daily_state` tensor
- Output: predicted future `60` day OHLC path
- Summary loss: `multi_horizon_ohlc` over `5/10/20/40/60` day OHLC-derived windows
- Value: `path_trade_value_v2` derived from predicted path
- Loss weights: `path=0.45 / summary=0.20 / value=0.20 / rank=0.15`
- TopK report: `1,3,5,10,20,50,100`
- Prediction mode: `compact`
- Early stopping: `patience=2`

## Explicit Comparison Profile

`all_channels_base_summary` is the legacy broad-TopK baseline. It keeps the old all-channel input surface and base 60-day summary loss, and remains useful because its test Top3/Top10 stayed stronger than the new default.

Command:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-legacy-all-channels-base --json
```

`summary_v2_multi_horizon_ohlc` is an explicit comparison profile, not the default mainline. It keeps:

- Store view: `daily_research/data/research_store/views/seq100_path60_todayclose_ohlcva.json`
- Model type: `gru_path_value`
- Output: predicted future `60` day OHLC path
- No symbol embedding, residual score, OHLCVA output, or richer target head

It changes `summary_loss_profile` from `base` to `multi_horizon_ohlc` and defaults `early_stopping_patience` to `2` in the narrow CLI. The summary loss constrains OHLC-derived summaries over available `5/10/20/40/60` day windows while preserving the same top-level loss weights; its implementation is vectorized but keeps the previous equal-weight per-horizon objective.

Command:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-summary-v2 --json
```

`summary_v2_no60` is a strict single-factor control for `summary_v2_multi_horizon_ohlc`, not a new summary family. It keeps the same OHLC-derived summary constraints, input channels, output path and loss weights, but removes only the full 60-day window, leaving `5/10/20/40`. New shape summaries belong to a future `summary_v3` experiment and must not be mixed into this control.

Command:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-summary-v2-no60 --json
```

`daily_only_no_minute` is an explicit input ablation profile, not the default mainline. It keeps the same labels, model type, output, loss weights, price anchor, split and TopK report, but changes `input_channel_profile` from `all` to `daily_only`. This removes `intraday_summary` and `limit_structure`, leaving `daily_raw + daily_state` as `32` input features. The narrow CLI defaults `early_stopping_patience` to `2`.

Command:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-daily-only --json
```

`daily_only_summary_v2` was promoted to the default research profile on 2026-07-08. The explicit command remains available for clarity and is equivalent to `train` unless overridden by flags.

Command:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-daily-only-summary-v2 --json
```

`no_intraday_summary` and `no_limit_structure` are partial input ablations. They keep labels, model type, output, loss weights, price anchor, split and TopK report aligned with the default, while removing exactly one minute-derived channel family.

Commands:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-no-intraday-summary --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-no-limit-structure --json
```

`direct_value_rank_5d/10d/60d` are explicit ranking comparison profiles, not the default mainline. They keep the same store view, input channels, split, price anchor and TopK report, but change `model_type` to `gru_direct_value`. The model emits only one scalar `score`; it does not emit future OHLC, path summary, OHLCVA, symbol embedding, residual score, or richer target outputs. Training derives `path_trade_value_v2_{horizon}d` from true future OHLC labels during loss calculation and uses `value=0.50 / rank=0.50` with path/summary/richer loss set to zero. The narrow CLI defaults `early_stopping_patience` to `2` and `batch_size` to `2048`.

Commands:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-direct-value-5d --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-direct-value-10d --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-direct-value-60d --json
```

## Concept Demotion

These surfaces are no longer default concepts:

- `alpha_v2`: historical infrastructure and evidence line.
- `path20`: historical or compatibility horizon.
- `symbol_embedding`: rejected as default until evidence changes.
- `residual_score`: comparison branch only.
- `richer_target`: comparison branch only.
- `ohlcva_unified`: paused branch.
- `rank_heavy_top1`: observation branch for narrow Top1 behavior.

These comparison surfaces remain named, but must not be mixed into the default profile:

- `table_path60_baseline`
- `path_only_next_open`
- `rank_heavy_top1`
- `all_channels_base_summary`
- `summary_v2_multi_horizon_ohlc`
- `summary_v2_no60`
- `daily_only_no_minute`
- `no_intraday_summary`
- `no_limit_structure`
- `direct_value_rank_5d`
- `direct_value_rank_10d`
- `direct_value_rank_60d`

The old phrase `LightGBM 191` is not a default baseline name because it mixes horizon, feature count, and model family. Use explicit names such as `table_path60_baseline`, `table_path20_191_baseline`, or `sequence_flat_lgbm_8400_sampled`.

## Evidence Boundary

Sequence path-value metrics are research evidence. `path_value_spread` means selected future path value relative to same-day universe average under the evaluation label. It is not live PnL and not an execution promotion claim.

This contract does not touch:

- `daily_research/output/active_execution_strategy.json`
- live/default/paper/broker state
- QDP active dataset pointers
- QDP provider ingest or data quality facts

## Next Method

- Next allowed action: use `daily_research.path_policy.seq100_mainline` for default train/contract/summarize operations.
- Next allowed action: run execution-layer backtest before any strategy or execution-candidate claim.
- Next allowed action: keep old branches as explicit comparison or archived evidence only.

The next default method after this slimming is `run_execution_layer_backtest()`:

```text
daily TopK
holding overlap
capital occupancy
T+1 sell constraint
limit-up/limit-down tradability
slippage and cost
exit rule
```

Until that execution-layer bridge exists, today-close path-only remains a research-ranking mainline, not a tradeable strategy.

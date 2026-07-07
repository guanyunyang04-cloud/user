# Path Policy Seq100 Mainline Slimming Contract

日期：`2026-07-07`

## Status

`process_complexity_slimmed / concept_surface_narrowed / research_only / no_active_change`

This contract turns the current seq100 path-value research into a narrow default work surface. It does not delete historical evidence, does not change QDP active data, and does not activate execution artifacts.

## Verdict

- Status: `process_complexity_slimmed / concept_surface_narrowed / research_only / no_active_change`.
- Mainline: `seq100_todayclose_path_only` is the only default seq100 path-value research surface.
- Active artifact impact: unchanged. This does not touch `daily_research/output/active_execution_strategy.json`, live/default, paper/live, broker, trade plans, QDP active dataset pointers, or QDP provider state.
- Concept decision: `alpha_v2`, `path20`, `symbol_embedding`, `residual_score`, `richer_target`, `ohlcva_unified`, and `rank_heavy_top1` are no longer default concepts.

## Default Mainline

The default research line is:

```text
research_store_view
-> seq100_x84_input
-> today_close_anchor
-> future60_ohlc_path
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
- Input: past `100 x 84` daily path/state tensor
- Output: predicted future `60` day OHLC path
- Value: `path_trade_value_v2` derived from predicted path
- Loss weights: `path=0.45 / summary=0.20 / value=0.20 / rank=0.15`
- TopK report: `1,3,5,10,20,50,100`
- Prediction mode: `compact`

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

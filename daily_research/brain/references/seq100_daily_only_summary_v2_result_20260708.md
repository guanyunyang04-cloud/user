# Seq100 Daily-Only Summary V2 Result

日期：`2026-07-08`

## Verdict

Status: `completed / explicit_combination_comparison / research_only / no_active_change`.

`daily_only_summary_v2` completed as an explicit combination profile. It does not replace `seq100_todayclose_path_only` as the default mainline.

The run shows that daily-only inputs and multi-horizon summary_v2 constraints can be combined cleanly, but their gains do not simply add. The profile improved validation versus `daily_only_no_minute` and kept high test rank IC, but test TopK path-value spread was weaker than both `daily_only_no_minute` Top1 and `summary_v2_no60` Top3/Top10.

Active artifact impact: unchanged. This does not touch `daily_research/output/active_execution_strategy.json`, live/default, paper/live, broker, trade plans, QDP active dataset pointers, or QDP provider state.

## Method

- Command: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-daily-only-summary-v2 --run-tag qdp_v2_seq100_path60_todayclose_daily_only_summary_v2_p2 --json`
- Run dir: `daily_research/output/path_policy/sequence_path_training/qdp_v2_seq100_path60_todayclose_daily_only_summary_v2_p2_20260708_193234`
- Generated at: `2026-07-08T20:32:39+08:00`
- Store view: `daily_research/data/research_store/views/seq100_path60_todayclose_ohlcva.json`
- Model type: `gru_path_value`
- Output: future `60 x 4` OHLC path
- Input profile: `daily_only`
- Input channels: `daily_raw`, `daily_state`
- Input dim: `32`
- Summary profile: `multi_horizon_ohlc`
- Summary windows: `5/10/20/40/60`
- Loss weights: `path=0.45 / summary=0.20 / value=0.20 / rank=0.15`
- Early stopping: `patience=2`
- Completed epochs: `3`; best epoch: `1`; device: `cuda`

Labels, train/validation/test split, price anchor, value target semantics and TopK report were kept aligned with the default today-close path-only mainline.

## Results

| model | split | rank IC | positive IC day rate | Top1 PV alpha | Top3 PV alpha | Top10 PV alpha |
|---|---|---:|---:|---:|---:|---:|
| base today-close path-only | validation | 0.1558 | 71.49% | 18.43% | 14.94% | 12.69% |
| daily-only no-minute | validation | 0.1438 | 71.07% | 18.87% | 16.65% | 14.93% |
| summary_v2 multi-horizon OHLC | validation | 0.1609 | 73.97% | 26.27% | 23.06% | 16.15% |
| daily-only summary_v2 | validation | 0.1555 | 73.14% | 21.98% | 19.64% | 16.03% |
| base today-close path-only | test | 0.1569 | 97.94% | 13.28% | 11.85% | 10.24% |
| daily-only no-minute | test | 0.1698 | 98.35% | 17.75% | 12.02% | 8.19% |
| summary_v2 multi-horizon OHLC | test | 0.1466 | 97.12% | 15.55% | 10.96% | 8.77% |
| daily-only summary_v2 | test | 0.1677 | 97.12% | 15.65% | 8.89% | 7.00% |

## Interpretation

- The combination improved validation IC from `0.1438` for `daily_only_no_minute` to `0.1555`, and validation TopK was strong. This supports that summary_v2 constraints are useful even without minute-derived inputs.
- Test IC stayed high at `0.1677`, close to `daily_only_no_minute` `0.1698` and above the default `0.1569`.
- Test Top1 fell from `daily_only_no_minute` `17.75%` to `15.65%`; Top3 and Top10 fell to `8.89%` and `7.00%`.
- The combination therefore did not produce additive gains. Daily-only seems helpful for broad IC, while summary_v2 helps validation and some narrow behavior, but together they did not repair broad TopK and weakened daily-only's strongest Top1 signal.
- This remains a useful comparison profile because it supports the concept-slimmed path `daily_raw + daily_state -> future60 OHLC path -> multi-horizon summary -> path value rank`, but it is not the current default replacement.

## Boundary

This is research evidence only. It does not change `daily_research/output/active_execution_strategy.json`, live/default, paper/live, broker state, trade plans, QDP active dataset pointers, or QDP provider state.

# Seq100 Input Ablation And Summary V2 No60 Result

日期：`2026-07-08`

## Verdict

Status: `completed / explicit_comparison / research_only / no_active_change`.

`no_intraday_summary`、`no_limit_structure` and `summary_v2_no60` completed as explicit comparison profiles. They do not replace `seq100_todayclose_path_only` as the default mainline.

The user's control-design objection was correct: `summary_v3_incremental` would both remove the duplicate 60-day window and add new summary constraints, so it is not a strict control experiment. That run was stopped and is not evidence. The formal third experiment is `summary_v2_no60`: it keeps the same summary_v2 OHLC-derived summary family and removes only the full-horizon 60-day window.

Active artifact impact: unchanged. This does not touch `daily_research/output/active_execution_strategy.json`, live/default, paper/live, broker, trade plans, QDP active dataset pointers, or QDP provider state.

## Method

Shared fixed contract:

- Store view: `daily_research/data/research_store/views/seq100_path60_todayclose_ohlcva.json`
- Model type: `gru_path_value`
- Output: future `60 x 4` OHLC path
- Price anchor: `today_close`
- Value column: `path_trade_value_v2_60d`
- Loss weights: `path=0.45 / summary=0.20 / value=0.20 / rank=0.15`
- Early stopping: `patience=2`
- TopK report: `1,3,5,10,20,50,100`

Completed runs:

| profile | command | run dir | best epoch | input dim | changed factor |
|---|---|---|---:|---:|---|
| `no_intraday_summary` | `train-no-intraday-summary --run-tag qdp_v2_seq100_path60_todayclose_no_intraday_summary_p2 --json` | `daily_research/output/path_policy/sequence_path_training/qdp_v2_seq100_path60_todayclose_no_intraday_summary_p2_20260708_091137` | 1 | 43 | remove `intraday_summary`; keep `daily_raw + daily_state + limit_structure` |
| `no_limit_structure` | `train-no-limit-structure --run-tag qdp_v2_seq100_path60_todayclose_no_limit_structure_p2 --json` | `daily_research/output/path_policy/sequence_path_training/qdp_v2_seq100_path60_todayclose_no_limit_structure_p2_20260708_101532` | 1 | 73 | remove `limit_structure`; keep `daily_raw + daily_state + intraday_summary` |
| `summary_v2_no60` | `train-summary-v2-no60 --run-tag qdp_v2_seq100_path60_todayclose_summary_v2_no60_p2 --json` | `daily_research/output/path_policy/sequence_path_training/qdp_v2_seq100_path60_todayclose_summary_v2_no60_p2_20260708_123559` | 1 | 84 | keep summary_v2 OHLC constraints, remove only the full 60-day window |

`summary_v3_incremental` run directory `daily_research/output/path_policy/sequence_path_training/qdp_v2_seq100_path60_todayclose_summary_v3_incremental_p2_20260708_114034` was interrupted and must not be used as evidence.

## Results

| model | split | rank IC | positive IC day rate | Top1 PV alpha | Top3 PV alpha | Top10 PV alpha |
|---|---|---:|---:|---:|---:|---:|
| base today-close path-only | validation | 0.1558 | 71.49% | 18.43% | 14.94% | 12.69% |
| daily-only no-minute | validation | 0.1438 | 71.07% | 18.87% | 16.65% | 14.93% |
| no intraday summary | validation | 0.1520 | 71.07% | 19.15% | 19.76% | 16.22% |
| no limit structure | validation | 0.1432 | 70.66% | 14.81% | 15.08% | 12.55% |
| summary_v2 multi-horizon OHLC | validation | 0.1609 | 73.97% | 26.27% | 23.06% | 16.15% |
| summary_v2 no60 | validation | 0.1535 | 69.01% | 14.41% | 16.91% | 14.99% |
| base today-close path-only | test | 0.1569 | 97.94% | 13.28% | 11.85% | 10.24% |
| daily-only no-minute | test | 0.1698 | 98.35% | 17.75% | 12.02% | 8.19% |
| no intraday summary | test | 0.1745 | 99.18% | 13.98% | 9.62% | 5.68% |
| no limit structure | test | 0.1482 | 99.18% | 15.59% | 8.45% | 5.61% |
| summary_v2 multi-horizon OHLC | test | 0.1466 | 97.12% | 15.55% | 10.96% | 8.77% |
| summary_v2 no60 | test | 0.1476 | 98.35% | 16.58% | 12.98% | 9.47% |

## Interpretation

Input ablation:

- Removing only `intraday_summary` produced the strongest test rank IC in this group: `0.1745`, higher than default `0.1569` and daily-only `0.1698`.
- Removing only `limit_structure` weakened test rank IC to `0.1482`, which suggests `limit_structure` carries more useful broad rank signal than the current `intraday_summary` family when the other minute-derived family remains.
- TopK behavior is not monotonic. Both partial-removal runs have weak test Top10 spreads around `5.6%`, while default all-channels remains stronger at `10.24%` and daily-only is `8.19%`.
- The safest conclusion is not "minute features are useless." Current minute-derived inputs are not necessary for test IC or narrow selection, but the two minute-derived families may interact for wider TopK breadth. This needs multi-seed or repeated split evidence before demoting the channels from all future experiments.

Summary loss:

- `summary_v2_no60` is a strict control against `summary_v2_multi_horizon_ohlc`: same OHLC-derived summary family, same loss weight, same input channels, but windows are `5/10/20/40` instead of `5/10/20/40/60`.
- Removing the full 60-day window reduced validation IC from `0.1609` to `0.1535`, so the full-horizon duplicate helped validation fit.
- On test, no60 slightly improved IC from `0.1466` to `0.1476` and improved Top1/Top3/Top10 from `15.55%/10.96%/8.77%` to `16.58%/12.98%/9.47%`.
- This supports the concern that adding 60-day OHLC-derived constraints on top of the existing 60-day base path summary can overweight full-horizon shape. Removing it improves test TopK versus summary_v2, but still does not clearly beat the default mainline on broad Top10.
- New shape summaries such as early return, repair, underwater area, roughness, time-to-hit or early-late contrast should be tested as a separate `summary_v3_new_features` experiment, not mixed into the no60 control.

## Boundary

This is research evidence only. It does not change `daily_research/output/active_execution_strategy.json`, live/default, paper/live, broker state, trade plans, QDP active dataset pointers, or QDP provider state.

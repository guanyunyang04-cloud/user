# Seq100 Direct Value Rank Result

日期：`2026-07-08`

## Verdict

`direct_value_rank_5d`、`direct_value_rank_10d` 和 `direct_value_rank_60d` 已完成为显式排序对照实验。它们不替代默认 `seq100_todayclose_path_only` 主线。

这组三个实验回答的是“序列模型能否不预测未来 OHLC 路径，而直接学习未来路径价值并排序”。结果支持这个方向具备可学习信号，尤其 60d direct ranker 在 test split 仍有正 IC 和稳定正日占比；但与默认 OHLC path-output 主线相比，60d direct ranker 的 test IC 与 TopK PV alpha 更弱，因此当前不应把默认输出从路径改成单一 score。

## Method

- Store view: `daily_research/data/research_store/views/seq100_path60_todayclose_ohlcva.json`
- Model type: `gru_direct_value`
- Input profile: `all`
- Input channels: `daily_raw`, `daily_state`, `intraday_summary`, `limit_structure`
- Input dim: `84`
- Output: one scalar `score`; no future OHLC path, no summary head, no richer target head
- Target: true future OHLC labels derived into `path_trade_value_v2_{horizon}d`
- Loss weights: `path=0.0 / summary=0.0 / richer=0.0 / value=0.5 / rank=0.5`
- Early stopping: `patience=2`
- Batch size: `2048`
- Device: `cuda`

Commands:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-direct-value-5d --run-tag qdp_v2_seq100_direct_value_5d_p2 --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-direct-value-10d --run-tag qdp_v2_seq100_direct_value_10d_p2 --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-direct-value-60d --run-tag qdp_v2_seq100_direct_value_60d_p2 --json
```

Run dirs:

- `daily_research/output/path_policy/sequence_path_training/qdp_v2_seq100_direct_value_5d_p2_20260708_012201`
- `daily_research/output/path_policy/sequence_path_training/qdp_v2_seq100_direct_value_10d_p2_20260708_024733`
- `daily_research/output/path_policy/sequence_path_training/qdp_v2_seq100_direct_value_60d_p2_20260708_040931`

## Results

| model | horizon value | best epoch | completed epochs | split | rank IC | positive IC day rate | Top1 PV alpha | Top3 PV alpha | Top10 PV alpha |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| direct value ranker | 5d | 2 | 4 | validation | 0.0662 | 62.81% | 2.48% | 1.82% | 1.46% |
| direct value ranker | 5d | 2 | 4 | test | 0.0424 | 62.55% | 0.002% | 0.30% | 0.60% |
| direct value ranker | 10d | 2 | 4 | validation | 0.0868 | 68.18% | 4.71% | 3.74% | 2.78% |
| direct value ranker | 10d | 2 | 4 | test | 0.0586 | 69.14% | 0.92% | 1.81% | 1.74% |
| direct value ranker | 60d | 1 | 3 | validation | 0.1435 | 70.66% | 21.75% | 18.42% | 14.38% |
| direct value ranker | 60d | 1 | 3 | test | 0.1025 | 92.59% | 9.46% | 7.94% | 7.17% |

Training histories:

| horizon | validation IC by epoch | early stop reason |
|---|---|---|
| 5d | 0.0530 -> 0.0662 -> 0.0571 -> 0.0602 | best epoch 2, then two non-improving epochs |
| 10d | 0.0804 -> 0.0868 -> 0.0725 -> 0.0632 | best epoch 2, then two non-improving epochs |
| 60d | 0.1435 -> 0.0804 -> 0.0872 | best epoch 1, then two non-improving epochs |

## Interpretation

- Direct score training is wired correctly: the model emits only `score`, path/summary/richer losses are zero, and value/rank losses use true future OHLC-derived `path_trade_value_v2_{horizon}d`.
- 10d is materially stronger than 5d on test IC and TopK PV alpha, so the very short 5d horizon is currently a weaker direct ranking target.
- 60d direct ranker has the strongest direct-ranker IC and TopK PV alpha, but it still underperforms the default 60d OHLC path-output mainline reported earlier: default test IC `0.1569`, Top1 `13.28%`, Top3 `11.85%`, Top10 `10.24%`; 60d direct test IC `0.1025`, Top1 `9.46%`, Top3 `7.94%`, Top10 `7.17%`.
- This supports keeping direct-value as a comparison surface. It does not support concept replacement from “predict path then derive value” to “directly score only”.
- The 60d run shows over-training quickly: train loss keeps falling while validation IC falls after epoch1. `patience=2` was useful and should remain the narrow default for these comparison profiles.

## Boundary

This is research evidence only. It does not change `daily_research/output/active_execution_strategy.json`, live/default, paper/live, broker state, trade plans, QDP active dataset pointers, or QDP provider state.

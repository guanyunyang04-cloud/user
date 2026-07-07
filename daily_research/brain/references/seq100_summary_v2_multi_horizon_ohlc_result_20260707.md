# Seq100 Summary V2 Multi-Horizon OHLC Result

日期：`2026-07-07`

## Verdict

`summary_v2_multi_horizon_ohlc` completed as an explicit comparison profile. It does not replace `seq100_todayclose_path_only` as the default mainline.

The experiment improved validation IC and validation TopK path-value spread, but test IC and broad test TopK were weaker than the original today-close path-only mainline. Test Top1 improved, so this profile is useful as a narrow Top1 observation branch, not as the default ranking model.

## Method

- Command: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-summary-v2 --run-tag qdp_v2_seq100_path60_todayclose_summary_v2 --json`
- Run dir: `daily_research/output/path_policy/sequence_path_training/qdp_v2_seq100_path60_todayclose_summary_v2_20260707_185027`
- Store view: `daily_research/data/research_store/views/seq100_path60_todayclose_ohlcva.json`
- Model type: `gru_path_value`
- Output: future `60 x 4` OHLC path
- Summary profile: `multi_horizon_ohlc`
- Loss weights: `path=0.45 / summary=0.20 / value=0.20 / rank=0.15`
- Completed epochs: `4`; best epoch: `1`; device: `cuda`

The comparison profile kept the concept surface slim: no symbol embedding, no residual score, no OHLCVA output, no richer future target head.

## Results

| model | split | rank IC | positive IC day rate | Top1 PV alpha | Top3 PV alpha | Top10 PV alpha |
|---|---|---:|---:|---:|---:|---:|
| base today-close path-only | validation | 0.1558 | 71.49% | 18.43% | 14.94% | 12.69% |
| summary_v2 multi-horizon OHLC | validation | 0.1609 | 73.97% | 26.27% | 23.06% | 16.15% |
| base today-close path-only | test | 0.1569 | 97.94% | 13.28% | 11.85% | 10.24% |
| summary_v2 multi-horizon OHLC | test | 0.1466 | 97.12% | 15.55% | 10.96% | 8.77% |

## Interpretation

- Validation improved across IC and TopK, so the added OHLC-derived constraints are learnable and not a wiring failure.
- Test Top1 improved from `13.28%` to `15.55%`, but Top3 and Top10 declined. This resembles mild concentration rather than a robust broad-rank improvement.
- Test IC declined from `0.1569` to `0.1466`, so default ranking quality is weaker than the original mainline.
- Because the model still outputs OHLC only, this result supports the idea that extra OHLC-derived constraints are a reasonable controlled comparison, but not enough to replace the current default.

## Boundary

This is research evidence only. It does not change `daily_research/output/active_execution_strategy.json`, live/default, paper/live, broker state, trade plans, QDP active dataset pointers, or QDP provider state.


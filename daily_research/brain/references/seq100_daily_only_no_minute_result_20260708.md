# Seq100 Daily-Only No-Minute Input Ablation Result

日期：`2026-07-08`

## Verdict

`daily_only_no_minute` completed as an explicit input ablation. It does not replace `seq100_todayclose_path_only` as the default mainline.

The result does not support the claim that minute-derived input channels are clearly necessary for daily path-value ranking. Removing `intraday_summary` and `limit_structure` improved test rank IC and narrow Top1/Top3 path-value spread, while weakening broad Top10 test spread. The most defensible conclusion is: current minute-derived channels are not a required signal source for the narrow selector, but they may still help wider TopK breadth or stability.

## Method

- Command: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-daily-only --run-tag qdp_v2_seq100_path60_todayclose_daily_only_no_minute_p2 --early-stopping-patience 2 --json`
- Run dir: `daily_research/output/path_policy/sequence_path_training/qdp_v2_seq100_path60_todayclose_daily_only_no_minute_p2_20260707_225758`
- Generated at: `2026-07-08T00:01:27+08:00`
- Store view: `daily_research/data/research_store/views/seq100_path60_todayclose_ohlcva.json`
- Model type: `gru_path_value`
- Output: future `60 x 4` OHLC path
- Input profile: `daily_only`
- Input channels: `daily_raw`, `daily_state`
- Input dim: `32` instead of the default `84`
- Removed input channels: `intraday_summary` (`41` dims) and `limit_structure` (`11` dims)
- Summary profile: `base`
- Loss weights: `path=0.45 / summary=0.20 / value=0.20 / rank=0.15`
- Early stopping: `patience=2`
- Completed epochs: `3`; best epoch: `1`; device: `cuda`

Labels, train/validation/test split, price anchor, loss weights, TopK report, and path-value target semantics were kept aligned with the default today-close path-only mainline.

## Results

| model | split | rank IC | positive IC day rate | Top1 PV alpha | Top3 PV alpha | Top10 PV alpha |
|---|---|---:|---:|---:|---:|---:|
| base today-close path-only | validation | 0.1558 | 71.49% | 18.43% | 14.94% | 12.69% |
| daily-only no-minute | validation | 0.1438 | 71.07% | 18.87% | 16.65% | 14.93% |
| base today-close path-only | test | 0.1569 | 97.94% | 13.28% | 11.85% | 10.24% |
| daily-only no-minute | test | 0.1698 | 98.35% | 17.75% | 12.02% | 8.19% |

## Interpretation

- Test rank IC improved from `0.1569` to `0.1698`, so the 32-dim daily-only input did not lose broad ranking signal on the test split.
- Test Top1 path-value spread improved from `13.28%` to `17.75%`; Top3 was roughly flat-to-better at `12.02%` versus `11.85%`.
- Test Top10 path-value spread fell from `10.24%` to `8.19%`, so minute-derived channels may still help wider selection breadth.
- Validation IC declined from `0.1558` to `0.1438`, but validation TopK spread improved. This mixed validation/test pattern argues against immediate default replacement.
- The cleanest next decomposition, if needed, is to separate `intraday_summary` removal from `limit_structure` removal. The current run removes both, matching the no-minute input question.

## Boundary

This is research evidence only. It does not change `daily_research/output/active_execution_strategy.json`, live/default, paper/live, broker state, trade plans, QDP active dataset pointers, or QDP provider state.

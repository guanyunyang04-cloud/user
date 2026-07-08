# Path Policy Seq100 Price Delta And Low OHLCVA Aux Result

Date: `2026-07-09`

## Verdict
- Status: `completed / explicit_comparison / research_only / no_active_change`.
- `daily_only_summary_v2_price_delta` completed with close-to-close log-delta price rhythm supervision. It improved validation/test IC versus the current default, but did not improve the narrow Top1 path-value spread.
- `daily_only_summary_v2_ohlcva_aux_low` completed with lower VA auxiliary weights. It produced the best validation IC among this comparison set and the strongest test Top1/Top3 path-value spread.
- `daily_only_summary_v2_ohlcva_aux_low_price_delta` completed, but the combined constraints were weaker than either single addition on test IC and TopK.
- Active artifact impact: unchanged. This did not touch `daily_research/output/active_execution_strategy.json`, live/default, broker state, trade plans, QDP active dataset pointers, or QDP provider state.

## Method
- Store view: `daily_research/data/research_store/views/seq100_path60_todayclose_ohlcva.json`
- Input profile: `daily_only` (`daily_raw + daily_state`, 32 dims)
- Anchor: `today_close`
- Default value semantics: price-only OHLC-derived `path_trade_value_v2_60d`
- Summary profile: `multi_horizon_ohlc`
- Early stopping: `patience=2`; all three new runs completed 3 epochs and selected best epoch 1.

### Run Tags
- `seq100_todayclose_path_only_daily_only_summary_v2_price_delta_20260708_234224`
- `seq100_todayclose_path_only_daily_only_summary_v2_ohlcva_aux_low_20260709_004303`
- `seq100_todayclose_path_only_daily_only_summary_v2_ohlcva_aux_low_price_delta_20260709_014409`

### Commands
```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-daily-only-summary-v2-price-delta --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-daily-only-summary-v2-ohlcva-aux-low --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-daily-only-summary-v2-ohlcva-aux-low-price-delta --json
```

## Loss Semantics
- `price_delta_loss`: converts predicted and true anchored close returns to `log(1 + close_ret)`, then applies SmoothL1 to close log first differences, including day 1 relative to the signal-day anchor. It is an auxiliary rhythm constraint on the OHLC path, not a replacement for path/value/rank loss.
- `va_level_loss`: SmoothL1 on auxiliary predicted volume/amount level dimensions.
- `va_delta_loss`: SmoothL1 on day-to-day volume/amount changes in the auxiliary path.
- In all OHLCVA auxiliary runs, `future_path` remains 4D OHLC, and summary/value/rank still use price-only `path_trade_value_v2_60d`.

## Results
Path-value alpha below is `alpha_path_trade_value_v2_60d`.

| profile | best epoch | val IC | test IC | test Top1 | test Top3 | test Top10 | val Top1 | val Top3 | val Top10 | test VA level MAE | test VA delta MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| default daily-only summary_v2 | 1 | 0.1555 | 0.1677 | 15.65% | 8.89% | 7.00% | 21.98% | 19.64% | 16.03% |  |  |
| OHLCVA aux high (`0.05/0.02`) | 1 | 0.1502 | 0.1459 | 22.06% | 12.76% | 8.13% | 21.95% | 19.64% | 14.86% | 0.5436 | 0.3064 |
| price_delta (`0.03`) | 1 | 0.1605 | 0.1781 | 15.41% | 11.36% | 7.81% | 19.27% | 14.95% | 13.46% |  |  |
| OHLCVA aux low (`0.02/0.01`) | 1 | 0.1626 | 0.1755 | 23.62% | 13.31% | 7.85% | 19.31% | 17.02% | 13.66% | 0.5499 | 0.3067 |
| OHLCVA aux low + price_delta | 1 | 0.1525 | 0.1527 | 18.06% | 11.09% | 6.91% | 21.99% | 17.79% | 14.98% | 0.5447 | 0.3068 |

## Interpretation
- `price_delta_loss` is useful for overall ranking: test IC improved from `0.1677` to `0.1781`. It does not by itself strengthen the very narrow Top1 selector.
- Low-weight VA auxiliary is the best single result in this batch for narrow selection: test Top1/Top3 PV alpha improved to `23.62%/13.31%`, while test IC stayed above the default at `0.1755`.
- High-weight VA had stronger narrow TopK than default but weaker IC. Lowering VA weights reduced the IC drag while preserving and improving narrow TopK.
- VA MAE alone is not the decision target. The low-VA run has slightly worse VA MAE than the high-VA run but better IC and TopK, so the useful effect is likely representation regularization /量价协同约束, not simply "more accurate VA forecast equals better rank".
- Combining low VA and price delta was not additive. It likely over-constrained the shared representation or shifted optimization toward competing auxiliary objectives.
- All three new profiles peaked at epoch 1 and then validation IC degraded, so `patience=2` remains justified for these narrow seq100 experiments.

## Current Judgment
- Do not auto-promote any new profile to default in this writeback.
- If the next decision prioritizes overall IC, `price_delta` is the cleanest improvement.
- If the next decision prioritizes narrow Top1/Top3 selection, `ohlcva_aux_low` is the strongest current candidate.
- A promotion decision should be explicit and should preferably run at least one repeat seed or a follow-up staged/lower-weight auxiliary experiment before replacing the default research entry.

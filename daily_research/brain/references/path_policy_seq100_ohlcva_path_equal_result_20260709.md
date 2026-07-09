# Path Policy Seq100 OHLCVA Equal Path-Loss Result

Date: `2026-07-09`

## Verdict
- Status: `completed / explicit_comparison / research_only / no_active_change`.
- `daily_only_summary_v2_ohlcva_path_equal` completed as the strict test requested: OHLCVA six fields enter the main `path_loss` equally, while summary/value/rank still use price-only OHLC-derived `path_trade_value_v2_60d`.
- Result: isolated test Top1 PV alpha improved to `26.20%`, the strongest current single-run Top1. However test IC fell to `0.1434`, and test Top3/Top10 were only `11.65%/6.34%`.
- Current judgment: keep as Top1-biased comparison evidence. It does not replace the default mainline, and it is less balanced than `daily_only_summary_v2_ohlcva_aux_low`.
- Active artifact impact: unchanged. This did not touch `daily_research/output/active_execution_strategy.json`, live/default, broker state, trade plans, QDP active dataset pointers, or QDP provider state.

## Method
- Command: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-daily-only-summary-v2-ohlcva-path-equal --json`
- Run dir: `daily_research/output/path_policy/sequence_path_training/seq100_todayclose_path_only_daily_only_summary_v2_ohlcva_path_equal_20260709_052341`
- Store view: `daily_research/data/research_store/views/seq100_path60_todayclose_ohlcva.json`
- Input profile: `daily_only` (`daily_raw + daily_state`, 32 dims)
- Model: `gru_ohlcva_aux_path_value`
- Path loss profile: `ohlcva_equal`
- Summary profile: `multi_horizon_ohlc`
- Loss weights: `path0.45 / summary0.20 / value0.20 / rank0.15 / price_delta0 / va_level0 / va_delta0`
- Early stopping: `patience=2`; completed 3 epochs; best epoch 1.

## Loss Semantics
- `future_ohlcva_aux_path` is a 6D OHLCVA output.
- `future_path` remains the first 4D OHLC slice and is still the only path used by summary/value/rank.
- Main `path_loss` is `mean(L_open, L_high, L_low, L_close, L_volume, L_amount)`, where each field loss is finite-target SmoothL1 over batch and days.
- `va_level_loss` and `va_delta_loss` are still reported as diagnostics, but their weights are zero in this profile, so VA is not added twice.

## Results
Path-value alpha below is `alpha_path_trade_value_v2_60d`.

| profile | best epoch | val IC | test IC | test Top1 | test Top3 | test Top10 | val Top1 | val Top3 | val Top10 | test VA level MAE | test VA delta MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| default daily-only summary_v2 | 1 | 0.1555 | 0.1677 | 15.65% | 8.89% | 7.00% | 21.98% | 19.64% | 16.03% |  |  |
| price_delta (`0.03`) | 1 | 0.1605 | 0.1781 | 15.41% | 11.36% | 7.81% | 19.27% | 14.95% | 13.46% |  |  |
| OHLCVA aux high (`0.05/0.02`) | 1 | 0.1502 | 0.1459 | 22.06% | 12.76% | 8.13% | 21.95% | 19.64% | 14.86% | 0.5436 | 0.3064 |
| OHLCVA aux low (`0.02/0.01`) | 1 | 0.1626 | 0.1755 | 23.62% | 13.31% | 7.85% | 19.31% | 17.02% | 13.66% | 0.5499 | 0.3067 |
| OHLCVA aux low + price_delta | 1 | 0.1525 | 0.1527 | 18.06% | 11.09% | 6.91% | 21.99% | 17.79% | 14.98% | 0.5447 | 0.3068 |
| OHLCVA equal path_loss | 1 | 0.1497 | 0.1434 | 26.20% | 11.65% | 6.34% | 19.37% | 18.30% | 15.11% | 0.5392 | 0.3059 |

## Training History
| epoch | train loss | path loss | value loss | rank loss | validation IC | best |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.1494 | 0.0917 | 0.0234 | 0.6796 | 0.1497 | yes |
| 2 | 0.1448 | 0.0880 | 0.0260 | 0.6559 | 0.0733 | no |
| 3 | 0.1248 | 0.0797 | 0.0394 | 0.5243 | 0.0594 | no |

## Interpretation
- The user's intuition that VA may matter for the very best rising candidate is supported by the test Top1 result: `26.20%` is higher than low-VA auxiliary's `23.62%`.
- The same experiment also shows that making VA equally part of the main reconstruction objective is not a balanced improvement. Validation IC degraded after epoch 1, final test IC was weaker than default, and test Top3/Top10 were weaker than low-VA auxiliary.
- The likely mechanism is optimization competition: keeping total path weight fixed while expanding from 4 to 6 fields reduces per-price-field reconstruction weight, while VA level has larger natural error scale. This can push the shared representation toward a sharper, Top1-biased signal and away from stable cross-sectional ranking.
- The cleaner current lesson is not "VA should be ignored"; it is "VA is useful as a lower-weight auxiliary/regularizer, while full equal-weight VA path reconstruction is too strong for the current objective."

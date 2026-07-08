# seq100 daily-only summary_v2 OHLCVA auxiliary result - 2026-07-08

## Setup
- Command: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-daily-only-summary-v2-ohlcva-aux --json`
- Run dir: `daily_research/output/path_policy/sequence_path_training/seq100_todayclose_path_only_daily_only_summary_v2_ohlcva_aux_20260708_221510`
- Store view: `daily_research/data/research_store/views/seq100_path60_todayclose_ohlcva.json`
- Input profile: `daily_only` (`daily_raw + daily_state`, 32 dims)
- Model: `gru_ohlcva_aux_path_value`
- Price path output used by value/rank: 4D OHLC (`path_dim=4`)
- Auxiliary output: 6D OHLCVA (`richer_path_dim=6`, `uses_ohlcva_aux_path=true`)
- Summary/value/rank semantics: price-only OHLC-derived `path_trade_value_v2_60d`
- Loss weights: `path0.45 / summary0.20 / value0.20 / rank0.15 / va_level0.05 / va_delta0.02`
- Early stopping: `patience=2`; completed 3 epochs; best epoch 1.

## Results
| split | IC | positive day rate | VA level MAE | VA delta MAE | Top1 PV alpha | Top3 PV alpha | Top10 PV alpha |
|---|---:|---:|---:|---:|---:|---:|---:|
| validation | 0.1502 | 73.14% | 0.5835 | 0.3003 | 21.95% | 19.64% | 14.86% |
| test | 0.1459 | 98.35% | 0.5436 | 0.3064 | 22.06% | 12.76% | 8.13% |

## Interpretation
- The implementation cleanly tests volume/amount auxiliary supervision without changing the ranking value definition: price-only `path_trade_value_v2_60d` remains the target for value/rank.
- Training loss and VA level loss decreased across epochs, but validation IC degraded after epoch 1, so best checkpoint remained epoch 1.
- Versus current default `daily_only_summary_v2` (`val IC 0.1555`, `test IC 0.1677`, test Top1/Top3/Top10 `15.65%/8.89%/7.00%`), OHLCVA auxiliary is weaker on IC but stronger on narrow TopK PV alpha, especially test Top1 and Top3.
- Current judgment: keep as comparison evidence. It does not replace the default mainline yet. The next focused follow-up would be lower VA weights, for example `va_level0.02 / va_delta0.01`, or VA auxiliary only after epoch 1 / via a separate head regularizer.

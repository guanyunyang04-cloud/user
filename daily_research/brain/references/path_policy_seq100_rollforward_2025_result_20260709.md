# Path Policy Seq100 Roll-Forward 2025 Result

Date: `2026-07-09`

## Verdict
- Status: `completed / roll_forward_test / research_only / no_active_change`.
- The old `2012-2023 train / 2024 validation / 2025 test` results should be treated as stale-train forward checks. They are useful, but they do not answer the user's production-like question: "after one more year of data is available, what happens if the model is retrained through 2024 and tested on 2025?"
- The formal roll-forward test in this reference uses `2012-2024` as train, refits normalization on train-year dates through 2024, fixes `epochs=1`, and reports 2025 `test` split metrics.
- Evaluation policy after this run: profile assessment should mainly pair the original 2024 validation evidence with the 2025 train-through-2024 forward test. The 2024 validation split remains the selection/stability anchor; the 2025 forward split is the recent production-like out-of-sample anchor.
- Current judgment: low-weight OHLCVA auxiliary is the best balanced roll-forward candidate because it has the strongest 2025 IC (`0.1927`). Equal OHLCVA path-loss remains Top1-biased: it has strongest Top1/Top10 but weakest IC.
- Active artifact impact: unchanged. This did not touch `daily_research/output/active_execution_strategy.json`, live/default, broker state, trade plans, QDP active dataset pointers, or QDP provider state.

## Data And Protocol
- Source view: `daily_research/data/research_store/views/seq100_path60_todayclose_ohlcva.json`
- Roll-forward view: `daily_research/data/research_store/views/seq100_path60_todayclose_ohlcva_train2012_2024_val2025_test2025.json`
- Roll-forward sample index: `daily_research/data/research_store/sample_index/seq100_path60_train2012_2024_val2025_test2025.parquet`
- Train samples: `5,708,975`
- 2025 validation/test samples: `689,446` each
- Normalization fit scope: `train_year_dates_only_2012_2024`
- Note: 2025 is duplicated as `validation` and `test` because the current trainer requires a validation split to save a checkpoint. All runs use `--epochs 1 --early-stopping-patience 0`, so 2025 is not used for multi-epoch checkpoint selection. The formal metric is the `test` split.

## Commands
All commands use the same store view and fixed epoch:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-daily-only-summary-v2 --store-view daily_research/data/research_store/views/seq100_path60_todayclose_ohlcva_train2012_2024_val2025_test2025.json --epochs 1 --early-stopping-patience 0 --run-tag rollforward_2025_daily_only_summary_v2_e1 --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-daily-only-summary-v2-price-delta --store-view daily_research/data/research_store/views/seq100_path60_todayclose_ohlcva_train2012_2024_val2025_test2025.json --epochs 1 --early-stopping-patience 0 --run-tag rollforward_2025_price_delta_e1 --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-daily-only-summary-v2-ohlcva-aux-low --store-view daily_research/data/research_store/views/seq100_path60_todayclose_ohlcva_train2012_2024_val2025_test2025.json --epochs 1 --early-stopping-patience 0 --run-tag rollforward_2025_ohlcva_aux_low_e1 --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-daily-only-summary-v2-ohlcva-aux-low-price-delta --store-view daily_research/data/research_store/views/seq100_path60_todayclose_ohlcva_train2012_2024_val2025_test2025.json --epochs 1 --early-stopping-patience 0 --run-tag rollforward_2025_ohlcva_aux_low_price_delta_e1 --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-daily-only-summary-v2-ohlcva-path-equal --store-view daily_research/data/research_store/views/seq100_path60_todayclose_ohlcva_train2012_2024_val2025_test2025.json --epochs 1 --early-stopping-patience 0 --run-tag rollforward_2025_ohlcva_path_equal_e1 --json
```

## Run Dirs
- `daily_research/output/path_policy/sequence_path_training/rollforward_2025_daily_only_summary_v2_e1_20260709_101340`
- `daily_research/output/path_policy/sequence_path_training/rollforward_2025_price_delta_e1_20260709_103947`
- `daily_research/output/path_policy/sequence_path_training/rollforward_2025_ohlcva_aux_low_e1_20260709_110647`
- `daily_research/output/path_policy/sequence_path_training/rollforward_2025_ohlcva_aux_low_price_delta_e1_20260709_113640`
- `daily_research/output/path_policy/sequence_path_training/rollforward_2025_ohlcva_path_equal_e1_20260709_120407`

## Results
Path-value alpha below is 2025 `test` split `alpha_path_trade_value_v2_60d`.

| profile | test IC | positive day rate | Top1 | Top3 | Top10 | VA level MAE | VA delta MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| default daily-only summary_v2 | 0.1846 | 96.71% | 18.44% | 10.09% | 6.96% |  |  |
| price_delta (`0.03`) | 0.1783 | 95.06% | 21.54% | 8.63% | 5.35% |  |  |
| OHLCVA aux low (`0.02/0.01`) | 0.1927 | 97.12% | 22.51% | 7.73% | 6.72% | 0.5564 | 0.3067 |
| OHLCVA aux low + price_delta | 0.1861 | 96.71% | 24.61% | 11.77% | 6.37% | 0.5496 | 0.3066 |
| OHLCVA equal path_loss | 0.1768 | 97.94% | 25.18% | 11.74% | 7.28% | 0.5451 | 0.3059 |

Machine-readable comparison:
- `daily_research/output/path_policy/sequence_path_training/rollforward_2025_profile_comparison_20260709.csv`
- `daily_research/output/path_policy/sequence_path_training/rollforward_2025_profile_comparison_20260709.json`

## Interpretation
- Adding 2024 to the training set materially changes the preferred reading. The default profile's 2025 IC rises to `0.1846`, so the old 2025 test result from a 2023-cutoff model was not the right proxy for the production-like case.
- Low-weight VA auxiliary is the best balanced profile in this roll-forward test. It improves IC over default by about `0.0081` while also improving Top1.
- `price_delta` is not additive here. It improves Top1 versus default but weakens IC, Top3 and Top10.
- Low VA plus price_delta is a stronger narrow Top1/Top3 candidate than low VA alone, but its IC is lower.
- Equal OHLCVA path-loss is still Top1-biased. It has strongest Top1 and Top10 but the weakest IC among the five profiles, supporting the previous judgment that full equal-weight VA reconstruction is too strong for the current ranking objective.
- No active/default or execution promotion follows from this run alone. The next defensible step would be either a multi-window roll-forward test, or an explicit decision about whether IC or very narrow Top1/Top3 selection is the primary research objective.

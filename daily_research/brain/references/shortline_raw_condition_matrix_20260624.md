# Shortline Raw Condition Matrix 20260624

## Verdict

- Status: `completed / research_only / shadow_only / no_active_change`.
- Run tag: `shortline_raw_condition_matrix_full_2012_2025_20260624_01`.
- Scope: after-close shortline conditional mechanism diagnostic over QDP `style_structural_alpha_v2_label_v2` sharded memmap unnormalized engineering features.
- Active artifact impact: unchanged. This run does not change `daily_research/output/active_execution_strategy.json`, live/default, paper/live/broker, score-backtest bridge, candidate matrix, execution-candidate state, or QDP registry pointers.
- Core verdict: previous single-feature diagnostics were enough to establish the broad anti-overheat / anti-chase direction, but not enough to decide whether a usable mechanism condition exists. This matrix tests train-fixed conditional combinations. The strongest conditional evidence is `pullback_intraday_recovery`; `market_confirmed_anti_overheat` and `industry_not_collapsing_not_overheat` are secondary context filters. Simple chase / mild-gap / late-strength conditions still do not stand as stable rules.

## Command

```powershell
C:\Users\ASUS\miniconda3\envs\yolos\python.exe -m daily_research.path_policy.shortline_raw_condition_matrix `
  --roles train,validation,test `
  --label-horizons 1,2,4 `
  --target-kinds net_abs,net_excess `
  --min-count-for-stability 500 `
  --run-tag shortline_raw_condition_matrix_full_2012_2025_20260624_01 `
  --output-root H:\quant_project\daily_research\output\path_policy\shortline_raw_condition_matrix\shortline_raw_condition_matrix_full_2012_2025_20260624_01 `
  --json
```

## Semantics

This diagnostic uses:

```text
QDP sharded memmap forecast_feature_store.dat
```

It does not use:

```text
normalized training pack
original OHLCV / original 5-minute bar lake tables
model predictions
topK strategy selection
```

The rule contract is intentionally conservative:

```text
1. Estimate numeric thresholds from train role only.
2. Freeze thresholds.
3. Apply the same conditions unchanged to validation and test.
4. Report D+1 next-open entry to future-open exits.
5. Report entry block/tradeability diagnostics, but do not pre-filter by future entry labels.
```

Label horizons:

```text
label_horizon=1: D+1 open entry -> D+2 open exit
label_horizon=2: D+1 open entry -> D+3 open exit
label_horizon=4: D+1 open entry -> D+5 open exit
```

Targets:

```text
net_abs
net_excess
round_trip_cost_bps = 20
```

## Input

Sharded memmap manifest:

```text
H:\quant_project\quant_data_platform\data\memmap\sharded\mainboard_style_structural_alpha_v2_label_v2_full_2010_2026_20260616_01\sharded_memmap_manifest.json
```

Scanned panel:

```text
shards: 154
stock-date rows: 8,971,715
features used by conditions: 22
conditions: 10
roles:
  train: 2012-2023
  validation: 2024
  test: 2025
```

Threshold feature coverage was recorded in:

```text
shortline_raw_condition_threshold_coverage.csv
```

The early-year `turn` missingness found during smoke did not break the full run because later train years provide enough finite observations. It remains explicitly visible in threshold coverage rather than hidden.

## Result Summary

Full run output summary:

```text
metric_rows: 180
monthly_rows: 9,204
stable_condition_rows: 60
validation_test_positive_net_rows: 9
all_roles_positive_lift_rows: 11
```

Best validation/test-positive candidates by stability:

| condition | target | exit | train_mean | val_mean | test_mean | min_lift | positive_lift_month_rate |
|---|---|---:|---:|---:|---:|---:|---:|
| pullback_intraday_recovery | net_excess | D+5_open | 0.0093% | 0.1417% | 0.3013% | 0.1126% | 0.585 |
| pullback_intraday_recovery | net_excess | D+3_open | -0.0598% | 0.1452% | 0.0974% | 0.0876% | 0.612 |
| pullback_intraday_recovery | net_abs | D+2_open | -0.1192% | 0.0806% | 0.0963% | 0.0355% | 0.715 |
| market_confirmed_anti_overheat | net_excess | D+5_open | -0.0382% | 0.0633% | 0.0119% | 0.0212% | 0.479 |
| industry_not_collapsing_not_overheat | net_abs | D+5_open | -0.0309% | 0.1310% | 0.4207% | 0.0023% | 0.375 |

Condition-level aggregate:

```text
pullback_intraday_recovery:
  validation/test positive rows: 5
  all-role positive-lift rows: 5
  avg selected_rate: 11.8%

market_confirmed_anti_overheat:
  validation/test positive rows: 3
  all-role positive-lift rows: 2
  avg selected_rate: 10.2%

industry_not_collapsing_not_overheat:
  validation/test positive rows: 1
  all-role positive-lift rows: 4
  avg selected_rate: 39.8%
```

Negative result:

```text
anti_overheat_core
low_mid_turn_mild_positive_gap
low_runup_high_before_low
mild_gap_late_low
anti_chase_late_strength
low_runup_late_intraday_low
quiet_vol_pullback
```

These did not produce validation/test-positive rows under the current stability filter. This is useful evidence against treating simple low-runup, simple mild gap, or simple late strength as sufficient standalone mechanisms.

## Interpretation

The result refines the prior single-feature conclusion:

```text
single-feature layer:
  high recent runup / high turnover / high relative runup are generally weak.

condition matrix:
  the useful version is not just "low runup".
  it is closer to "pullback or drawdown state plus intraday repair/close-position confirmation",
  with market or industry context acting as secondary filters.
```

This still does not constitute a tradable strategy:

```text
selected_rate is broad, around 10% for the strongest pullback condition.
monthly worst cases remain large.
topK selection among selected stocks is not solved.
entry price range / conditional intraday fill is not modeled.
exit policy is still fixed future-open diagnostic.
```

Therefore the correct next step is not promotion or active execution. The next step is to use these condition families as priors for a supervised shortline scorer or a narrower condition-grid refinement:

```text
1. scorer baseline: train a simple/interpretable shortline upside scorer on raw condition families and stable raw features.
2. decile audit: require monotonicity or at least top-decile lift over validation/test.
3. topK audit: only after score decile works, test top1/top3/top5 with train-selected rule and validation/test holdout.
4. entry label v1: separately build conditional price-zone entry labels if we want non-fixed open buying.
```

## Artifacts

Root:

```text
H:\quant_project\daily_research\output\path_policy\shortline_raw_condition_matrix\shortline_raw_condition_matrix_full_2012_2025_20260624_01
```

Files:

```text
shortline_raw_condition_thresholds.csv
shortline_raw_condition_threshold_coverage.csv
shortline_raw_condition_definitions.csv
shortline_raw_condition_metrics.csv
shortline_raw_condition_monthly.csv
shortline_raw_condition_stable_candidates.csv
shortline_raw_condition_matrix_report.json
shortline_raw_condition_matrix_report.md
```

## Implementation

Added:

```text
daily_research/path_policy/shortline_raw_condition_matrix.py
daily_research/path_policy/tests/test_shortline_raw_condition_matrix.py
```

The implementation includes:

```text
train-only threshold estimation
validation/test frozen-threshold application
condition definitions export
overall metrics
monthly metrics
stable candidate table
entry tradeability / limit-up-block diagnostics
threshold finite coverage report
```

## Verification

Focused tests:

```text
pytest daily_research/path_policy/tests/test_shortline_raw_condition_matrix.py -q
2 passed
```

Smoke over real sharded memmap:

```text
shortline_raw_condition_matrix_smoke_20260624_01
roles: train=2023, validation=2024, test=2025
max_shards_per_role: 1
scanned_rows: 218,100
status: completed
```

Full run:

```text
shortline_raw_condition_matrix_full_2012_2025_20260624_01
scanned_shards: 154
scanned_rows: 8,971,715
status: completed
```

## Next Allowed Actions

- Use `pullback_intraday_recovery`, `market_confirmed_anti_overheat`, and `industry_not_collapsing_not_overheat` as candidate priors for the next shortline scorer.
- Add stricter subconditions around pullback repair, because the current selected set is still too broad for topK execution.
- Evaluate score deciles and topK only after the scorer is trained or condition grid is narrowed.
- Do not promote, bridge, candidate-matrix, or touch active/default/live artifacts from this evidence alone.

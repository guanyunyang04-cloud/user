# Shortline Raw Upside Diagnostic 20260624

## Verdict

- Status: `completed / research_only / shadow_only / no_active_change`.
- Run tag: `shortline_raw_upside_diagnostic_full_2012_2025_20260624_01`.
- Scope: after-close shortline upside mechanism diagnostic over QDP `style_structural_alpha_v2_label_v2` sharded memmap unnormalized engineering features.
- Active artifact impact: unchanged. This run does not change `daily_research/output/active_execution_strategy.json`, live/default, paper/live/broker, score-backtest bridge, candidate matrix, execution-candidate state, or QDP registry pointers.
- Core verdict: the raw/unnormalized engineering-feature layer confirms the previous normalized-pack signal direction: under fixed D+1 next-open entry labels, stable single-feature evidence is still mostly anti-overheat / anti-chase. High turnover, high 10d/20d runup, and high relative runup versus industry tend to be weaker; some intraday path-shape variables remain candidate conditioning signals, but are not yet stable enough alone to become rules.

## Command

```powershell
C:\Users\ASUS\miniconda3\envs\yolos\python.exe -m daily_research.path_policy.shortline_raw_upside_diagnostic `
  --roles train,validation,test `
  --label-horizons 1,2,4 `
  --target-kinds net_abs,net_excess `
  --run-tag shortline_raw_upside_diagnostic_full_2012_2025_20260624_01 `
  --output-root H:\quant_project\daily_research\output\path_policy\shortline_raw_upside_diagnostic\shortline_raw_upside_diagnostic_full_2012_2025_20260624_01 `
  --json
```

## Semantics

This diagnostic uses:

```text
QDP sharded memmap forecast_feature_store.dat
```

not:

```text
QDP normalized training pack
```

and not yet:

```text
deep raw OHLCV / original 5-minute bar lake tables
```

Therefore it is the current cheapest interpretable layer:

```text
raw data lake facts
  -> QDP unnormalized engineered features      <-- this run
  -> normalized training pack                  <-- previous 307-feature scan
  -> model tensors
```

Labels are still T+1-aware fixed next-open labels:

```text
label_horizon=1: D+1 open entry -> D+2 open exit
label_horizon=2: D+1 open entry -> D+3 open exit
label_horizon=4: D+1 open entry -> D+5 open exit
```

Cost:

```text
round_trip_cost_bps = 20
target_kind = net_abs / net_excess
```

This is feature mechanism profiling, not model training, not topK strategy selection, and not executable entry-zone research.

## Input

Sharded memmap manifest:

```text
H:\quant_project\quant_data_platform\data\memmap\sharded\mainboard_style_structural_alpha_v2_label_v2_full_2010_2026_20260616_01\sharded_memmap_manifest.json
```

Scanned panel:

```text
shards: 154
stock-date rows: 8,971,715
features: 34 selected mechanism features
roles:
  train: 2012-2023
  validation: 2024
  test: 2025
```

Feature groups:

```text
recent returns: ret_1d / ret_3d / ret_5d / ret_10d / ret_20d
volatility / turnover: vol_5d / vol_20d / turn / turn_z20
raw daily bar shape: gap / range / body / shadow / limit-like flags
position / drawdown: distance_to_20d_high / distance_to_20d_low / local_drawdown_20d
intraday summaries: first/last 30m, close-to-vwap, close position, low-time, high-before-low, range, realized vol
market / industry context: market breadth, industry excess, stock-minus-industry returns
```

## Stable Same-Sign Result

Stable same-sign means `pearson_corr_raw` has the same sign across:

```text
train / validation / test
```

for a given:

```text
target_kind + label_horizon + feature
```

Summary:

```text
metrics rows: 612
bin rows: 5,484
stable same-sign rows: 85
common stable features across all 6 target/horizon combos: 7
```

Common stable raw/unnormalized features:

| feature | sign | mean_abs_corr |
|---|---:|---:|
| turn | neg | 0.03198 |
| ret_10d | neg | 0.03028 |
| stock_ret_20_minus_industry | neg | 0.02935 |
| ret_20d | neg | 0.02784 |
| stock_ret_5_minus_industry | neg | 0.02581 |
| distance_to_20d_high | neg | 0.02565 |
| local_drawdown_20d | neg | 0.02565 |

Interpretation:

```text
high turnover is weaker;
high recent 10d/20d runup is weaker;
high runup relative to industry is weaker;
high position close to 20d high is weaker under this fixed next-open label.
```

This reinforces the normalized-pack conclusion that the first stable shortline signal is not simple strong-continuation chasing.

## Test-Set Bin Examples

`ret_3d`, `test / net_abs / D+2_open`:

```text
lowest bin:  feature_mean -7.65%, target +0.1302%, hit 50.20%
highest bin: feature_mean +9.98%, target -0.2575%, hit 42.42%
```

`ret_20d`, `test / net_abs / D+2_open`:

```text
lowest bin:  feature_mean -16.66%, target +0.2051%, hit 51.68%
highest bin: feature_mean +27.68%, target -0.2466%, hit 42.72%
```

`turn`, `test / net_abs / D+2_open`:

```text
extreme high turnover bin: feature_mean 15.54, target -0.2583%, hit 42.59%
```

These examples should be read as descriptive feature evidence, not as a standalone strategy.

## Candidate Path-Shape Signals

Some intraday path-shape variables are worth follow-up, especially:

```text
intraday_low_time_frac
intraday_high_before_low
raw_open_gap_1d
```

Examples in `test / net_abs / D+2_open`:

```text
intraday_low_time_frac highest bin: target +0.1482%, hit 51.81%
intraday_high_before_low category-like high side: less negative than low side
raw_open_gap_1d: non-monotonic; large positive gap is not automatically bad in raw-bin view
```

But these are not common stable across all train/validation/test and all target/horizon combinations. They are candidates for conditional analysis, not direct rules.

## Implementation

Added:

```text
daily_research/path_policy/shortline_raw_upside_diagnostic.py
daily_research/path_policy/tests/test_shortline_raw_upside_diagnostic.py
```

The diagnostic reads each selected QDP shard:

```text
forecast_feature_store.dat
labels/cumulative_return_1to20
labels/cumulative_excess_return_1to20
```

and outputs:

```text
raw feature metrics
raw feature bins
leaderboard
stable train/validation/test same-sign correlations
common stable correlations across all target/horizon combos
JSON/Markdown report
```

Important reproducibility correction: stable correlation CSV outputs are now part of the script contract, not only a one-off post-run analysis.

## Artifacts

Root:

```text
H:\quant_project\daily_research\output\path_policy\shortline_raw_upside_diagnostic\shortline_raw_upside_diagnostic_full_2012_2025_20260624_01
```

Files:

```text
shortline_raw_upside_feature_metrics.csv
shortline_raw_upside_feature_bins.csv
shortline_raw_upside_feature_leaderboard.csv
shortline_raw_upside_stable_corr_features.csv
shortline_raw_upside_common_stable_corr_features.csv
shortline_raw_upside_diagnostic_report.json
shortline_raw_upside_diagnostic_report.md
```

## Verification

Focused contract test:

```text
pytest daily_research/path_policy/tests/test_shortline_raw_upside_diagnostic.py -q
2 passed
```

Smoke over real sharded memmap:

```text
shortline_raw_upside_diagnostic_smoke_20260624_01
roles: train,validation,test
max_shards_per_role: 1
scanned_rows: 217,428
feature_count: 34
status: completed
```

Full run:

```text
shortline_raw_upside_diagnostic_full_2012_2025_20260624_01
scanned_shards: 154
scanned_rows: 8,971,715
feature_count: 34
status: completed
```

## Next Step

The next useful step is conditional feature interaction, not immediate topK strategy selection:

```text
shortline_raw_condition_matrix_v1
```

Suggested conditions:

```text
low recent runup + late intraday low
low/medium turnover + mild positive gap
anti-overheat + market breadth regime
stock not overheated relative to industry + industry not collapsing
intraday path-shape signals conditional on low 10d/20d runup
```

If true original OHLCV / 5-minute thresholds are required, build a raw feature sidecar or read data-lake source tables directly. Do not infer raw executable thresholds from normalized training pack values.

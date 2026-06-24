# Shortline Upside Feature Profile 20260624

## Verdict

- Status: `completed / research_only / shadow_only / no_active_change`.
- Run tag: `shortline_upside_feature_profile_full_2012_2025_20260624_01`.
- Scope: full-history 307-feature profile for after-close shortline upside potential over the existing QDP alpha_v2 label_v2 training pack.
- Active artifact impact: unchanged. This run does not change `daily_research/output/active_execution_strategy.json`, live/default, paper/live/broker, score-backtest bridge, candidate matrix, execution-candidate state, or QDP registry pointers.
- Verdict: the current fixed D+1-open shortline label profile does not support a simple "chase the strongest stock" story. Across train/validation/test, the more stable single-feature relations are mostly `low_is_good`: lower turnover, lower recent 5d/20d return, lower volatility, lower amount/turnover rank, lower intraday range, and lower relative runup. A smaller set of `high_is_good` timing/shape features exists, such as price position from local peak, low occurring later in the day, high before low, mild open gap, and market amount expansion.

## Command

```powershell
C:\Users\ASUS\miniconda3\envs\yolos\python.exe -m daily_research.path_policy.shortline_upside_feature_profile `
  --roles train,validation,test `
  --label-horizons 1,2,4 `
  --target-kinds net_abs,net_excess `
  --group-scopes all,role,year,role_year `
  --run-tag shortline_upside_feature_profile_full_2012_2025_20260624_01 `
  --output-root H:\quant_project\daily_research\output\path_policy\shortline_upside_feature_profile\shortline_upside_feature_profile_full_2012_2025_20260624_01 `
  --json
```

## Semantics

This is not topK strategy selection and not model training. It asks:

```text
Given D-close information, which feature ranks describe future shortline upside potential?
```

Feature statistics use same-date cross-sectional percentile ranks across all 307 alpha_v2 features. Labels are T+1-aware open-to-open fixed-entry labels:

```text
label_horizon=1: D+1 open entry -> D+2 open exit
label_horizon=2: D+1 open entry -> D+3 open exit
label_horizon=4: D+1 open entry -> D+5 open exit
```

Costs:

```text
round_trip_cost_bps = 20
target_kind = net_abs / net_excess
```

Do not treat D+1 open -> D+1 close as executable under A-share T+1.

## Input

Training pack:

```text
H:\quant_project\quant_data_platform\data\memmap\training_pack\mainboard_style_structural_alpha_v2_label_v2_pack_full_2012_2025_20260616_01\qdp_training_pack_manifest.json
```

Selected samples:

```text
rows: 7,421,714
date groups: 3,337
features: 307
train: 2012-2023, 6,050,268 rows, 2,894 dates
validation: 2024, 681,979 rows, 221 dates
test: 2025, 689,467 rows, 222 dates
```

## Base Target Shape

Role-level target means after 20bps round-trip cost:

```text
train net_abs:
  D+2_open -0.1646%, hit 45.51%
  D+3_open -0.1300%, hit 46.42%
  D+5_open -0.0640%, hit 47.54%

validation net_abs:
  D+2_open -0.1331%, hit 45.34%
  D+3_open -0.0711%, hit 45.13%
  D+5_open +0.0872%, hit 45.95%

test net_abs:
  D+2_open -0.0696%, hit 45.43%
  D+3_open +0.0735%, hit 47.41%
  D+5_open +0.3628%, hit 49.71%

train net_excess:
  D+2_open -0.1833%, hit 42.83%
  D+3_open -0.1656%, hit 44.23%
  D+5_open -0.1287%, hit 45.56%

validation net_excess:
  D+2_open -0.2182%, hit 43.94%
  D+3_open -0.2392%, hit 44.90%
  D+5_open -0.2707%, hit 46.19%

test net_excess:
  D+2_open -0.1520%, hit 41.87%
  D+3_open -0.0975%, hit 43.44%
  D+5_open +0.0081%, hit 44.47%
```

Interpretation: the baseline fixed-entry universe is weak after cost, especially on excess return. Any shortline selector must materially separate top candidates from the base universe.

## Stable Feature Counts

Stable means the feature has the same rank-IC direction and the same top-decile vs bottom-decile spread direction across train, validation, and test.

```text
net_abs h1:    181 stable, 13 high_is_good, 168 low_is_good
net_abs h2:    183 stable, 14 high_is_good, 169 low_is_good
net_abs h4:    181 stable, 18 high_is_good, 163 low_is_good
net_excess h1: 184 stable, 14 high_is_good, 170 low_is_good
net_excess h2: 184 stable, 14 high_is_good, 170 low_is_good
net_excess h4: 177 stable, 19 high_is_good, 158 low_is_good
```

154 features were stable in the same direction across all six target/horizon combinations. Among these, 142 were `low_is_good` and only 12 were `high_is_good`.

## Top Stable Low-Is-Good Signals

For `net_abs` / D+5_open, the strongest common stable signals are:

```text
cs_rank_ret_20d
cs_rank_amount_20d
cs_z_ret_20d
local_ret_20d
ret_20d
cs_rank_turn
turn
cs_z_turn
peer_adv_bucket_ret_20d_rank
relative_to_peer_adv_ret_20d
local_vol_20d
vol_20d
cs_z_intraday_intraday_range
stock_ret_20_minus_industry
cs_rank_intraday_intraday_range
distance_to_20d_low / local_distance_to_low_20d
relative_to_peer_adv_ret_5d
peer_adv_bucket_ret_5d_rank
raw_intraday_range_1d
```

Meaning: under the current fixed next-open shortline label, lower recent runup, lower turnover/amount rank, lower volatility, lower intraday range, and lower relative strength tend to describe better future shortline returns. This is closer to shortline mean-reversion / anti-overheat behavior than pure strong-continuation.

## Top Stable High-Is-Good Signals

The strongest common stable `high_is_good` signals are smaller but interpretable:

```text
price_from_local_peak
raw_low_from_prev_close_1d
cs_rank_intraday_high_before_low
cs_z_intraday_high_before_low
cs_rank_intraday_low_time_frac
cs_z_intraday_low_time_frac
market_amount_expansion_share
intraday_low_time_frac
intraday_open_gap / raw_open_gap_1d / cs_rank_intraday_open_gap
```

Meaning: after-close shortline upside appears more associated with intraday path shape, mild open-gap/low-position timing, and market amount expansion than with high turnover or high recent momentum alone.

## Interpretation

The first important research correction is:

```text
Do not start by tuning topK/holding period.
Start by defining and profiling stock-date upside potential.
```

The second correction is:

```text
The stable single-feature evidence currently favors "avoid overheated/high-turnover/high-volatility recent winners"
more than "buy the strongest continuation".
```

This does not prove a profitable strategy. It provides the feature-level map needed before building a scorer.

## Artifacts

```text
feature metrics:
H:\quant_project\daily_research\output\path_policy\shortline_upside_feature_profile\shortline_upside_feature_profile_full_2012_2025_20260624_01\shortline_upside_feature_metrics.csv

decile profile:
H:\quant_project\daily_research\output\path_policy\shortline_upside_feature_profile\shortline_upside_feature_profile_full_2012_2025_20260624_01\shortline_upside_decile_profile.csv

target summary:
H:\quant_project\daily_research\output\path_policy\shortline_upside_feature_profile\shortline_upside_feature_profile_full_2012_2025_20260624_01\shortline_upside_target_summary.csv

all feature leaderboard:
H:\quant_project\daily_research\output\path_policy\shortline_upside_feature_profile\shortline_upside_feature_profile_full_2012_2025_20260624_01\shortline_upside_feature_leaderboard.csv

stable same-direction features:
H:\quant_project\daily_research\output\path_policy\shortline_upside_feature_profile\shortline_upside_feature_profile_full_2012_2025_20260624_01\shortline_upside_stable_same_direction_features.csv

common stable features across all target/horizon combinations:
H:\quant_project\daily_research\output\path_policy\shortline_upside_feature_profile\shortline_upside_feature_profile_full_2012_2025_20260624_01\shortline_upside_common_stable_features_all_targets.csv

stable feature counts:
H:\quant_project\daily_research\output\path_policy\shortline_upside_feature_profile\shortline_upside_feature_profile_full_2012_2025_20260624_01\shortline_upside_stable_feature_counts.csv

report JSON:
H:\quant_project\daily_research\output\path_policy\shortline_upside_feature_profile\shortline_upside_feature_profile_full_2012_2025_20260624_01\shortline_upside_feature_profile_report.json
```

## Next Step

Recommended next step:

```text
Build a supervised shortline upside scorer baseline from stable feature groups:
  low-overheat / low-turnover / low-volatility / low-recent-runup group,
  intraday path-shape group,
  market/industry confirmation group.

Evaluate score deciles first.
Only after score monotonicity and top-decile expectation are proven should topK, entry-zone, and exit policy be tuned.
```

Do not promote or deploy anything from this profile. Do not rebuild the full QDP memmap yet. A derived shortline entry-label pack is still reserved for conditional intraday entry research.

# Shortline Upside Feature Semantics 20260624

## Verdict

- Status: `completed / research_only / shadow_only / no_active_change`.
- Run tag: `shortline_upside_feature_semantics_full_2012_2025_20260624_01`.
- Purpose: add type-aware semantics to the full 307-feature shortline upside profile so `high_is_good` / `low_is_good` is not misread as universal economic high/low meaning.
- Active impact: unchanged. No `active_execution_strategy.json`, live/default, paper/live/broker, candidate matrix, score-backtest bridge, or QDP registry pointer changed.
- Core correction: the QDP training pack feature panel is `features_are_normalized=true`, `method=zscore`, `fit_role=train_only`. Therefore categorical/bucket outputs group standardized training features, not raw category ids. For raw flag/bucket interpretation, inspect pre-normalized feature sources or build a raw-feature sidecar.

## Command

```powershell
C:\Users\ASUS\miniconda3\envs\yolos\python.exe -m daily_research.path_policy.shortline_upside_feature_profile `
  --roles train,validation,test `
  --label-horizons 1,2,4 `
  --target-kinds net_abs,net_excess `
  --group-scopes all,role,year,role_year `
  --run-tag shortline_upside_feature_semantics_full_2012_2025_20260624_01 `
  --output-root H:\quant_project\daily_research\output\path_policy\shortline_upside_feature_profile\shortline_upside_feature_semantics_full_2012_2025_20260624_01 `
  --json
```

The first foreground command timed out at 30 minutes but the process continued and completed. Final progress:

```text
processed_groups: 3337 / 3337
processed_rows: 7,421,714 / 7,421,714
```

## Implementation

Updated:

```text
daily_research/path_policy/shortline_upside_feature_profile.py
daily_research/path_policy/tests/test_shortline_upside_feature_profile.py
```

New outputs added to the prior profile contract:

```text
shortline_upside_feature_semantics.csv
shortline_upside_categorical_profile.csv
shortline_upside_categorical_summary.csv
shortline_upside_market_level_profile.csv
```

Semantics classes:

```text
continuous_ordinal
already_ranked
zscore_continuous
market_level
binary_flag
ordinal_bucket
membership_flag
ordinal_count
```

Preferred interpretation:

```text
continuous / already-ranked / zscore:
  rank IC + decile are valid numeric-rank scans.

binary / membership:
  use group comparison; do not interpret as graded high/low.

bucket:
  use bucket group profile; high/low is safe only if bucket ordering is contractual.

market-level:
  use date-level relation across regimes; same-date cross-sectional rank is weak or misleading.
```

## Full Feature Type Counts

```text
continuous_ordinal    142
already_ranked         68
zscore_continuous      63
market_level           15
binary_flag            12
ordinal_bucket          3
membership_flag         2
ordinal_count           2
```

Source families:

```text
intraday                   127
other                       31
daily_price_volume          27
cross_section_transform     21
valuation                   20
industry                    20
market                      15
local_path                  14
raw_daily_bar               14
peer                        12
event_quality                4
index_membership             2
```

## Stable Feature Result

Stable means same direction across train / validation / test for both rank IC sign and top-vs-bottom decile spread sign.

```text
stable rows: 1108
common stable features across all six target/horizon combos: 154
```

Counts by target/horizon:

```text
net_abs h1:    16 high_is_good, 170 low_is_good
net_abs h2:    17 high_is_good, 169 low_is_good
net_abs h4:    21 high_is_good, 163 low_is_good
net_excess h1: 17 high_is_good, 171 low_is_good
net_excess h2: 14 high_is_good, 170 low_is_good
net_excess h4: 22 high_is_good, 158 low_is_good
```

Common stable features by direction/type:

```text
high_is_good:
  already_ranked         3
  continuous_ordinal     5
  market_level           1
  zscore_continuous      3

low_is_good:
  already_ranked        39
  continuous_ordinal    68
  zscore_continuous     35
```

Important: no `binary_flag`, `membership_flag`, or `ordinal_bucket` feature survived as common stable across all six target/horizon combinations.

## Interpretation

The previous full-feature profile conclusion remains mostly unchanged, but its interpretation is now cleaner:

```text
Stable shortline single-feature evidence is dominated by ordered continuous/rank/zscore features,
not by raw binary flags or bucket ids.
```

The strongest common stable signals are still mostly anti-overheat:

```text
low cs_rank_amount_20d
low cs_rank_turn / cs_z_turn / turn
low 20d return / local 20d return / cs_rank_ret_20d / cs_z_ret_20d
low distance_to_20d_low / local_distance_to_low_20d
low vol_5d / local_vol_5d / vol_20d / local_vol_20d
low intraday range / realized-vol style features
low peer/industry relative runup
```

This means the full-history, fixed D+1-open baseline still does not support a simple "chase the hottest/most active stock" story.

## Categorical Caveat

The type-aware categorical output is useful as a diagnostic, but not yet enough for raw economic interpretation because the QDP training pack stores normalized features:

```text
features_are_normalized: true
normalization: train-only zscore
dtype: float16
```

Therefore values such as `1.982422` in `local_high_volatility_flag` are standardized feature values, not raw flag value `1`. Do not write rules like:

```text
if local_high_volatility_flag == 1 then ...
```

from this normalized pack output.

For executable or human-readable rule mining, next step should use either:

```text
pre-normalized QDP feature source, or
raw-feature diagnostic sidecar, or
manifest normalization inverse map where safe.
```

## Market-Level Caveat

Market-level features now have a separate date-level profile. They should not be read primarily through same-day cross-sectional deciles, because many are date-context variables rather than stock-specific variables.

Examples with larger date-level correlations appear in validation/test subsets, but these are regime diagnostics, not direct stock selection rules.

## Artifacts

```text
root:
H:\quant_project\daily_research\output\path_policy\shortline_upside_feature_profile\shortline_upside_feature_semantics_full_2012_2025_20260624_01

report:
shortline_upside_feature_profile_report.json
shortline_upside_feature_profile_report.md

feature semantics:
shortline_upside_feature_semantics.csv

numeric profile:
shortline_upside_feature_metrics.csv
shortline_upside_decile_profile.csv
shortline_upside_feature_leaderboard.csv

type-aware profile:
shortline_upside_categorical_profile.csv
shortline_upside_categorical_summary.csv
shortline_upside_market_level_profile.csv

stable features:
shortline_upside_stable_same_direction_features.csv
shortline_upside_common_stable_features_all_targets.csv
shortline_upside_stable_feature_counts.csv
```

## Verification

```text
pytest daily_research/path_policy/tests/test_shortline_upside_feature_profile.py -q
4 passed
```

Additional smoke:

```text
shortline_upside_feature_semantics_smoke_20260624_02
shortline_upside_feature_semantics_smoke_20260624_03
```

## Next Step

Use the type-aware profile to build the first supervised/statistical `shortline_upside_score_v1`, but only from interpretable groups:

```text
continuous/rank/zscore stable low-overheat group
intraday path-shape stable group
market-regime context as gating/context, not direct cross-sectional rank
raw flag/bucket features only after raw-value sidecar or inverse-normalized audit
```

Still do not convert this feature profile into a topK strategy, candidate matrix, bridge, active artifact, paper/live signal, or promotion evidence.

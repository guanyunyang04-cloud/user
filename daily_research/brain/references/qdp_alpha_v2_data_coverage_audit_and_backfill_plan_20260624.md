# QDP Alpha V2 Data Coverage Audit And Backfill Plan 20260624

## Verdict

- Status: `completed_audit / backfill_plan_ready / research_only / no_active_change`.
- Run tag: `qdp_alpha_v2_feature_coverage_audit_20260624_02`.
- Scope: full finite-coverage audit of the current QDP `style_structural_alpha_v2_label_v2` sharded memmap.
- Active impact: unchanged. No `active_execution_strategy.json`, live/default, paper/live/broker, QDP registry pointer, canonical bundle pointer, memmap pointer, or training pack pointer was changed.
- Core verdict: the current alpha_v2 memmap is usable as historical research evidence but should not remain the future canonical default after data backfill. The main data-base gaps are turnover context, valuation context, and one intraday derived z-score anomaly.

## Command

```powershell
$env:PYTHONPATH='H:\quant_project\quant_data_platform\src;H:\quant_project'
C:\Users\ASUS\miniconda3\envs\yolos\python.exe -m quant_data_platform.cli audit-sharded-feature-coverage `
  --manifest-json H:\quant_project\quant_data_platform\data\memmap\sharded\mainboard_style_structural_alpha_v2_label_v2_full_2010_2026_20260616_01\sharded_memmap_manifest.json `
  --output-root H:\quant_project\daily_research\output\path_policy\data_coverage_audit `
  --run-tag qdp_alpha_v2_feature_coverage_audit_20260624_02 `
  --json
```

## Tooling Added

Added:

```text
quant_data_platform/src/quant_data_platform/memmap/coverage_audit.py
quant_data_platform/tests/test_feature_coverage_audit.py
```

CLI:

```text
qdp audit-sharded-feature-coverage
```

The audit is read-only:

```text
registry_impact: unchanged
memmap_impact: unchanged
active_artifact_impact: unchanged
```

## Input

Manifest:

```text
H:\quant_project\quant_data_platform\data\memmap\sharded\mainboard_style_structural_alpha_v2_label_v2_full_2010_2026_20260616_01\sharded_memmap_manifest.json
```

Scan summary:

```text
feature_count: 307
scanned_shards: 183
scanned_rows: 10,175,900
years: 2010-2026
feature_profile: style_structural_alpha_v2
canonical_dataset_id: policy_input_bundle__f926a496f69c61f6b92b5faf
```

## Findings

Suspicious features:

```text
total: 36
turnover_context: 7
valuation_context: 28
intraday_context: 1
```

Key findings:

```text
turnover_context_has_historical_coverage_gap
valuation_context_has_historical_or_field_specific_coverage_gap
cs_z_intraday_last_5m_ret_has_derived_feature_coverage_anomaly
```

### Turnover Context Gap

Affected fields:

```text
turn
turn_z20
turn_ratio_5_20
cs_rank_turn
cs_z_turn
industry_rank_turn
industry_z_turn
```

Coverage shape:

```text
2010-2015: zero coverage
2016 onward: usable
2024-2025: ~99% coverage
```

Interpretation:

```text
This is not a complete OHLCV failure. Price/return/volatility fields have coverage in early years.
The missing chain is turnover or its required base fields, likely raw turn / circulating shares / float shares / float market cap.
```

Backfill priority: `P0`.

### Valuation Context Gap

Affected families:

```text
valuation_peTTM_*
valuation_pbMRQ_*
valuation_psTTM_*
valuation_pcfNcfTTM_*
```

Coverage shape:

```text
2010-2014: near-zero coverage
2015: partial
2016 onward: mostly usable, but pcfNcfTTM remains materially sparse
2024-2025: pe/pb/ps good, pcfNcfTTM still only ~40%-50%
```

Interpretation:

```text
Part of this is likely data-base incompleteness.
Part of pcfNcfTTM may be naturally sparse or provider-specific.
Valuation backfill must be PIT-safe; do not use future-disclosed fundamentals to fill earlier dates.
```

Backfill priority: `P2`.

### Intraday Derived Anomaly

Affected field:

```text
cs_z_intraday_last_5m_ret
```

Coverage shape:

```text
recent_2024_2025_mean_finite_rate: 0
intraday_context overall mean coverage: high
```

Interpretation:

```text
This is probably not missing raw 5-minute data.
It is more likely a derived cross-sectional z-score construction, index alignment, constant-column handling, or naming/mapping bug.
```

Backfill priority: `P1`, but as a code/feature-generation repair rather than raw data backfill.

### Healthy Groups

Groups with broadly acceptable coverage:

```text
raw_kline
market_context
sector_relative_context
adjust_context
index_context
peer_context
local_state_context
intraday_context except cs_z_intraday_last_5m_ret
```

This means the current data base is incomplete in specific domains, not globally broken.

## Decision

Future work should treat the current alpha_v2 memmap as a historical incomplete-version artifact:

```text
usable for recorded research evidence
not ideal as future canonical default
not worth preserving for forward compatibility after a correct replacement exists
```

But old artifacts should be cleaned only after a replacement exists:

```text
1. backfill canonical data
2. generate new sharded memmap
3. validate new coverage
4. build new training pack / sidecars as needed
5. run minimal A/B checks
6. switch registry pointers
7. then clean old incomplete memmap/pack artifacts
```

Do not delete old memmap/pack before replacement and registry switch, because recorded evidence still needs lineage.

## Backfill Plan

### P0: Turnover / Liquidity Base Fields

Goal:

```text
complete turnover-related base data for 2010-2015 and verify 2016+ consistency
```

Fields:

```text
raw turn / turnover_rate if provider supplies it
volume
amount
circulating_share / float_share
float_market_cap
total_market_cap
```

Rules:

```text
prefer provider raw turn when available
derive turn = volume / circulating_share only when units and share base are verified
record source, unit, provider, fetch time, and PIT date semantics
do not patch memmap directly
```

Validation:

```text
turn finite coverage by year
turn range sanity
turn vs volume/float_share consistency
split/adjust-factor independence
halt/suspension behavior
cross-provider sample diff if multiple providers exist
```

### P1: Intraday Derived Feature Repair

Goal:

```text
repair cs_z_intraday_last_5m_ret coverage anomaly
```

Checks:

```text
raw intraday_last_5m_ret coverage
cross-sectional z-score implementation
constant day handling
date/symbol alignment
column naming in feature projection
```

Expected outcome:

```text
no raw data backfill unless raw intraday_last_5m_ret itself is missing
feature-generation fix plus regenerated memmap
```

### P2: Valuation Backfill

Goal:

```text
improve valuation availability without introducing disclosure-time leakage
```

Fields:

```text
peTTM
pbMRQ
psTTM
pcfNcfTTM
market cap fields used to derive them
report disclosure / effective date if fundamentals are used
```

Rules:

```text
PIT first; if PIT cannot be guaranteed, mark as non-PIT or lag conservatively
pcfNcfTTM may remain sparse; do not force-fill with unsafe values
missing flags remain useful and should be kept
```

### P3: Regenerate Derived Artifacts

After P0-P2:

```text
new canonical bundle id
new sharded memmap id
new training pack id
new date-slate/structured sidecars as needed
```

The replacement should be named as a new data product, not an in-place mutation.

Suggested naming:

```text
style_structural_alpha_v2_label_v2_backfilled_202606xx
```

### P4: Cleanup Policy

Only after replacement is validated:

```text
mark old alpha_v2 memmap/training packs deprecated
update active QDP registry pointers to new version
write brain evidence with replacement pointer
then remove old incomplete acceleration artifacts if disk pressure or simplicity requires it
```

Clean only generated acceleration artifacts, not canonical raw/canonical lake lineage.

## Artifacts

Output root:

```text
H:\quant_project\daily_research\output\path_policy\data_coverage_audit\qdp_alpha_v2_feature_coverage_audit_20260624_02
```

Files:

```text
coverage_audit_report.json
feature_year_finite_rates.csv
feature_coverage_summary.csv
group_year_finite_rates.csv
group_coverage_summary.csv
suspicious_feature_coverage.csv
```

## Verification

Focused tests:

```text
pytest quant_data_platform/tests/test_feature_coverage_audit.py -q
2 passed
```

Compile:

```text
py_compile coverage_audit.py cli.py
passed
```

## Next Allowed Actions

1. Inspect canonical/QDP lake tables and provider adapters for turnover, circulating shares, and valuation field availability.
2. Implement P0 turnover/liquidity backfill first.
3. Repair `cs_z_intraday_last_5m_ret` feature-generation anomaly.
4. Plan PIT-safe valuation backfill separately.
5. Regenerate sharded memmap only after source-domain backfills pass coverage and sanity checks.
6. Do not clean old memmap/training-pack artifacts until a validated replacement and registry switch exist.

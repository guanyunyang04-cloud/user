# QDP Alpha V2 Data Base Repair 20260624

## Verdict

- Status: `partial_code_repair_completed / research_only / no_active_change`.
- Scope: implement the first data-base repair patches identified after `qdp_alpha_v2_feature_coverage_audit_20260624_02`.
- Active impact: unchanged. No `daily_research/output/active_execution_strategy.json`, live/default, paper/live/broker, QDP registry pointer, canonical bundle pointer, memmap pointer, or training pack pointer was changed.
- Core verdict: two code-level causes were repaired, but the existing alpha_v2 sharded memmap and training pack remain stale until regenerated.

## Repairs

### P0/P2 Loader Schema Repair: Legacy Valuation Aliases

Files:

```text
daily_research/data_lake/policy_input_loader.py
daily_research/data_lake/tests/test_import_traditional_baostock_v2_snapshot.py
```

Change:

```text
turnover_rate -> turn
pe / pe_ttm -> peTTM
pb / pb_mrq -> pbMRQ
ps / ps_ttm -> psTTM
pcf_ncf_ttm -> pcfNcfTTM
```

Interpretation:

```text
This repairs a loader/schema mismatch for legacy valuation sidecars.
It is not a complete historical data backfill.
It only helps when the underlying sidecar actually contains those legacy alias columns.
If early years have no valuation rows or no turnover source at all, canonical source backfill is still required.
```

### P1 Intraday Derived Feature Repair: `last_5m_ret`

Files:

```text
daily_research/data_platform/build_intraday_daily_features.py
daily_research/data_platform/contracts.py
daily_research/data_platform/tests/test_domain_contract.py
daily_research/data_platform/tests/test_import_external_quant_zip.py
```

Root cause:

```text
The current raw 5-minute source uses a 15:00:00 closing point bar where open == close.
The old definition last_5m_ret = last_bar.close / last_bar.open - 1 therefore collapsed to 0.
That made cs_z_intraday_last_5m_ret all-NaN after cross-sectional z-score because the base column was constant.
```

Fix:

```text
last_5m_ret = last_bar.close / previous_bar.close - 1
```

This preserves the intended semantic signal: the last interval's close-to-close move into the final close.

The intraday daily feature dataset identity now also records:

```text
feature_contract_version = intraday_daily_features_v2
last_5m_ret_policy = last_close_to_previous_5m_close_return
```

This prevents regenerated sidecars from being semantically confused with old sidecars that used the collapsed `last.close / last.open - 1` definition.

The `build_intraday_daily_features --years` option now also filters source 5m shards by shard date bounds. Previously it was only written into the target spec, which made partial-year rebuild plans ambiguous.

## Live Sample Validation

Recomputed the first 2025 raw intraday shard through the repaired fast path:

```text
rows: 872
finite last_5m_ret: 872
unique rounded to 8dp: 596
nonzero: 603
std: about 0.00203
```

This confirms the repair restores non-trivial variance at the source-derived daily intraday feature level.

## P0 Source Coverage Audit After Alias Repair

The canonical valuation sidecar currently referenced by QDP is:

```text
data_platform_valuation__13151c89bf9f57eda45ed76e
```

Its raw schema is mixed by historical source period:

```text
2010-2015: turnover_rate, pe, pb
2016-2026: turn, peTTM, pbMRQ, psTTM, pcfNcfTTM
```

Full sidecar scan:

```text
rows: 10,855,197
shards: 2,017
```

Key yearly finite coverage:

```text
2010 turnover_rate: 95.58%, pe: 100.00%, pb: 99.98%
2011 turnover_rate: 95.50%, pe: 100.00%, pb: 99.98%
2012 turnover_rate: 96.58%, pe: 100.00%, pb: 100.00%
2013 turnover_rate: 96.12%, pe: 100.00%, pb: 100.00%
2014 turnover_rate: 92.04%, pe: 100.00%, pb: 99.95%
2015 turnover_rate: 85.86%, pe: 100.00%, pb: 100.00%
2016-2026 turn: about 91.56%-99.76%, peTTM/pbMRQ/psTTM/pcfNcfTTM: present in current source schema
```

Small canonical-bundle loader smoke for `2010-01-04` to `2010-01-29` after the alias repair:

```text
universe: 1,531
dates: 20
turn finite: 29,750 / 30,620, rate 97.16%
peTTM finite: 30,280 / 30,620, rate 98.89%
pbMRQ finite: 30,260 / 30,620, rate 98.82%
psTTM: missing
pcfNcfTTM: missing
```

Interpretation:

```text
P0 turnover_context 2010-2015 zero coverage in the old memmap was primarily a schema/loader mismatch.
It should be repaired by regenerating features through the fixed loader.
P2 valuation early-year pe/pb should also materially improve after regeneration.
P2 psTTM/pcfNcfTTM early-year coverage is still a real source gap or unimported source gap and should remain missing/PIT-safe unless a proper source is added.
```

## Verification

Focused tests:

```powershell
$env:PYTHONPATH='H:\quant_project\quant_data_platform\src;H:\quant_project'
C:\Users\ASUS\miniconda3\envs\yolos\python.exe -m pytest `
  H:\quant_project\daily_research\data_lake\tests\test_import_traditional_baostock_v2_snapshot.py `
  H:\quant_project\daily_research\data_platform\tests\test_domain_contract.py `
  H:\quant_project\daily_research\data_platform\tests\test_import_external_quant_zip.py `
  -q
```

Result:

```text
18 passed
```

## Boundaries

Unchanged:

```text
daily_research/output/active_execution_strategy.json
QDP registry pointers
canonical bundle pointers
current sharded memmap pointers
current training pack pointers
old generated artifacts
```

Infrastructure added for safe replacement builds:

```text
qdp build-sharded-memmap --canonical-dataset-id <explicit_policy_input_bundle_id>
```

This allows building a replacement sharded memmap from a newly generated policy bundle without first changing QDP `root_manifest.json`.

Important:

```text
The existing alpha_v2 sharded memmap still contains the old generated values.
The repair only affects newly generated sidecars/memmaps/training packs.
Do not treat old shortline or alpha_v2 results as if this patch retroactively fixed their inputs.
```

## Next Allowed Actions

1. Audit whether legacy `turnover_rate -> turn` coverage is enough for 2010-2015, or whether circulating-share/float-share source backfill remains required.
2. Regenerate intraday daily feature sidecars with the repaired `last_5m_ret` definition.
3. Rebuild a new alpha_v2-style sharded memmap/training pack as a new data product, not an in-place mutation.
4. Run the coverage audit on the replacement memmap and specifically verify:

```text
turn / turnover_context coverage by year
valuation_peTTM / pbMRQ / psTTM / pcfNcfTTM coverage by year
intraday_last_5m_ret variance
cs_z_intraday_last_5m_ret finite coverage
```

5. Only after replacement validation should registry pointers be switched and old incomplete generated artifacts cleaned.

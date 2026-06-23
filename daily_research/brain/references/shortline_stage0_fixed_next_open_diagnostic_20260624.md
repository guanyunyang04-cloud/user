# Shortline Stage 0 Fixed Next-Open Diagnostic 20260624

## Verdict

- Status: `completed / research_only / shadow_only / no_active_change`.
- Run tag: `shortline_stage0_full_vt_20260624_01`.
- Scope: Stage 0 after-close shortline mechanism diagnostic over the existing QDP alpha_v2 label_v2 training pack.
- Active artifact impact: unchanged. This run does not change `daily_research/output/active_execution_strategy.json`, live/default, paper/live/broker, score-backtest bridge, candidate matrix, execution-candidate state, or QDP registry pointers.
- Verdict: the three hand-built shortline mechanism scores did not pass validation-to-test robustness under the fixed D+1 open entry baseline. This is useful negative evidence, not a deployable strategy.

## Command

```powershell
C:\Users\ASUS\miniconda3\envs\yolos\python.exe -m daily_research.path_policy.shortline_stage0_diagnostics `
  --roles validation,test `
  --top-ks 1,3,5,10 `
  --label-horizons 1,2,4 `
  --run-tag shortline_stage0_full_vt_20260624_01 `
  --output-root H:\quant_project\daily_research\output\path_policy\shortline_stage0\shortline_stage0_full_vt_20260624_01 `
  --json
```

## Semantics

- Signal time: D after close.
- Entry baseline: fixed D+1 open, with `entry_tradeable`, `entry_limit_up_buy_blocked`, and `entry_suspended_or_no_open` filtering.
- T+1 exits:
  - `label_horizon=1`: D+1 open entry -> D+2 open exit.
  - `label_horizon=2`: D+1 open entry -> D+3 open exit.
  - `label_horizon=4`: D+1 open entry -> D+5 open exit.
- Do not treat D+1 open -> D+1 close as executable under A-share T+1.
- Round-trip cost: `20 bps`.
- Mechanism features: 22 selected normalized alpha_v2 features, converted to same-date cross-sectional ranks. Missing mechanism features: none.

## Input

Training pack:

```text
H:\quant_project\quant_data_platform\data\memmap\training_pack\mainboard_style_structural_alpha_v2_label_v2_pack_full_2012_2025_20260616_01\qdp_training_pack_manifest.json
```

Selected samples:

```text
roles: validation,test
rows: 1,371,446
date groups: 443
validation dates: 221
test dates: 222
```

Mechanisms:

```text
strong_continuation
trend_pullback
industry_spread
combined_shortline
```

## Main Result

Validation best by net absolute return:

```text
industry_spread / top1 / D+2_open
validation net_abs:      +0.1401%
validation net_excess:   +0.0552%
validation positive months: 50.0%
validation worst month:  -1.7573%
```

Same candidate on test:

```text
industry_spread / top1 / D+2_open
test net_abs:      -0.4389%
test net_excess:   -0.5206%
test positive months: 33.3%
test worst month:  -1.8086%
```

This fails the required validation-selected same-candidate test check.

Test-only best by net absolute return was:

```text
trend_pullback / top1 / D+5_open
test net_abs:      +0.4045%
test net_excess:   -0.0373%
test positive months: 50.0%
test worst month:  -3.7470%
```

This is diagnostic only. It cannot be used as a formal selection result because it is selected on test.

## Interpretation

The existing QDP pack is sufficient for fixed next-open Stage 0 diagnostics, and the Stage 0 code path is now functional. However, the naive hand-built mechanisms are not yet strong enough:

```text
validation signal is weak;
same-candidate test does not hold;
net excess is mostly negative after cost;
monthly stability is poor;
top1 variants are noisy;
topK broadening does not repair the expectation.
```

The result does not disprove after-close shortline research. It does show that simple rank-weighted technical recipes over current normalized features should be treated as baseline diagnostics, not as candidate production logic.

## Artifacts

```text
daily CSV:
H:\quant_project\daily_research\output\path_policy\shortline_stage0\shortline_stage0_full_vt_20260624_01\shortline_stage0_daily.csv

monthly CSV:
H:\quant_project\daily_research\output\path_policy\shortline_stage0\shortline_stage0_full_vt_20260624_01\shortline_stage0_monthly.csv

summary CSV:
H:\quant_project\daily_research\output\path_policy\shortline_stage0\shortline_stage0_full_vt_20260624_01\shortline_stage0_summary.csv

report JSON:
H:\quant_project\daily_research\output\path_policy\shortline_stage0\shortline_stage0_full_vt_20260624_01\shortline_stage0_report.json
```

## Next Step

Do not promote these rules. The next useful research step is to move from hand-weighted mechanism recipes toward either:

```text
1. conditional entry label/scorer research:
   gap filter, high-open no-chase, low-open breakdown block, entry-zone fill;

2. supervised shortline alpha scorer:
   still using fixed executable exits first, but learning the score from labels instead of hand weights;

3. mechanism-specific audits:
   isolate continuation, pullback, and industry spread by market regime, open gap, liquidity, and blocked-entry buckets.
```

Do not rebuild the full QDP memmap yet. A derived `shortline_entry_label_v1` pack becomes justified only when the next experiment explicitly needs conditional intraday entry labels.

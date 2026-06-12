# Expanded Factor Smoke And Personal Gate Scope Fix

Canonical brain source: `traditional_quant_research/brain/references/research_log/2026-06-04_expanded_factor_smoke_and_gate_scope_fix.md`.


## Summary

Run `frontier_personal_protocol_grid_20260604_055649` is the first real diagnostic smoke for `factor_set=expanded`.

It used Baostock-only expanded factors, `2026` single-year evaluation, `20d/monthly/top_n=100/buffer=3.0`, `30 bps`, `100m` capital stress, `10 bps per 1 pct` participation impact, and personal capital `1m`.

The run confirms the expanded factor path can pass through combined constraint, merged evidence, personal gate, and protocol ledger. It does not create a `personal_backtest_candidate`, `personal_paper_candidate`, or `strategy_candidate`.

## Result

- `factor_set`: `expanded`
- `years`: `2026`
- `top_n_values`: `[100]`
- `evidence_scopes`: `diagnostic_relaxed_personal_gate`
- `personal_backtest_candidate_count`: `0`
- `personal_paper_candidate_count`: `0`
- `strategy_candidate_count`: `0`
- `decision`: `keep_personal_research_backtest_only`
- Best diagnostic row: `top100_multifactor_low_corr_rank_score_baseline_penalty_1`
- Best diagnostic mean annualized return: `0.115833`
- Best diagnostic `total_periods`: `2`
- Best diagnostic `eval_year_count`: `1`
- Best diagnostic failed gates: `formal_profile_gate,walk_forward_gate,sample_gate`

## Interpretation

The positive 2026 single-year return is useful as a pipeline smoke and ranking hint, not as strategy evidence. The sample has only `2` completed monthly periods and only one evaluation year, so it cannot satisfy walk-forward coverage or formal sample gates.

The best row being `multifactor_low_corr_rank_score` is directionally interesting because the prior formal `top_n=20/50/100` core-factor grid failed. However, this run is too small to decide whether expanded factors improve 2017-2026 weak years. The next useful experiment is a formal core-vs-expanded 2017-2026 comparison, likely starting with `top_n=100/200` rather than `20/50`.

## Governance Fix

The smoke exposed a personal gate labeling issue: a single-year smoke using formal threshold defaults was initially labeled `formal_personal_backtest_candidate_gate` even though it failed walk-forward and sample coverage. The gate logic was fixed so `formal_profile_gate` and `evidence_scope` require both formal threshold settings and actual evidence coverage. Short or incomplete evidence now becomes `diagnostic_relaxed_personal_gate` even if thresholds are not relaxed.

Regression test added:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest traditional_quant_research/tests/test_frontier_personal_candidate_gate.py -q
```

The targeted test passes and the smoke was rerun with `--resume-run-dir`; its summary and ledger now correctly report `diagnostic_relaxed_personal_gate`.

## Next Step

Run a formal 2017-2026 expanded-factor protocol comparison after choosing the cost/top-N grid. Current evidence suggests using `top_n=100/200` and preserving `core` as the comparability control.

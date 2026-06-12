# Personal Protocol Gate Governance

## Summary

This update strengthens the Baostock-only personal research ladder by separating formal personal candidate evidence from relaxed diagnostic smoke evidence.

The main rule is:

- `formal_personal_backtest_candidate_gate` can produce `personal_backtest_candidate` and `start_paper_tracking`.
- `diagnostic_relaxed_personal_gate` can only produce `personal_research/backtest_only`, even when the short-window return looks strong.

This protects the current North Star: efficient personal quant research, without letting one-year smoke tests or threshold overrides pollute the candidate ledger.

## Code Changes

- `frontier_personal_candidate_gate.py`
  - Added `FORMAL_PERSONAL_GATE_SCOPE` and `DIAGNOSTIC_PERSONAL_GATE_SCOPE`.
  - Added formal profile inference across year coverage, period count, return, positive-year, weak-year, drawdown, exposure, fee, impact, and capital assumptions.
  - Added `formal_profile_gate`, `evidence_scope`, and `gate_profile_detail` to gate rows.
  - Relaxed profile runs now fail `formal_profile_gate` and remain `personal_research/backtest_only`.

- `low_corr_frontier_combined_constraint_audit.py`
  - Added `top_n_values`.
  - Reuses each built evaluation panel across multiple Top-N values.
  - Carries `top_n` into summary, trades, liquidity, exposure, industry exposure, metadata, and aggregate outputs.

- `frontier_personal_protocol_grid.py`
  - Runs combined constraint by eval year, writing `personal_protocol_grid_progress.csv`.
  - Merges yearly combined outputs into a standard combined evidence directory.
  - Runs the personal gate once on merged multi-Top-N evidence.
  - Carries `formal_profile_gate`, `evidence_scope`, and `gate_profile_detail` into `personal_protocol_grid_ledger.csv`.

- `frontier_personal_paper_tracking_bootstrap.py`
  - Adds `top_n` to candidate IDs so `top20` and `top50` variants do not collide.

## Real Smoke

Run:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m traditional_quant_research.experiments.frontier_personal_protocol_grid --top-n-values 20 --years 2026 --signals multifactor_low_corr_rank_score --signal-penalty-strengths multifactor_low_corr_rank_score=1.0 --final-end-date 2026-06-01 --required-start-year 2026 --required-end-year 2026 --min-eval-year-count 1 --min-total-periods 1 --min-positive-year-rate 0 --min-mean-annualized-return -999 --min-weakest-year-annualized-return -999 --max-worst-drawdown -1.0
```

Output run:

- `frontier_personal_protocol_grid_20260604_025606`
- `best_mean_annualized_return`: about `0.284307`
- `evidence_scopes`: `diagnostic_relaxed_personal_gate`
- `personal_backtest_candidate_count`: `0`
- `strategy_candidate_count`: `0`
- decision: `keep_personal_research_backtest_only`

Interpretation: the smoke proves the link is runnable and observable, not that the strategy is a formal candidate.

## Current Research State

The existing formal `personal_backtest_candidate` remains the previous 2017-2026 rolling IC candidate from `frontier_personal_candidate_gate_20260604_004311`.

No new `personal_backtest_candidate`, `personal_paper_candidate`, or `strategy_candidate` is added by this update.

The next research step is to run the formal 2017-2026 personal protocol grid for `top_n=20/50/100` under the default gate profile.

## Verification

- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest traditional_quant_research/tests -q`
  - `231 passed`
- `git diff --check -- traditional_quant_research`
  - passed
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.doc_guard check`
  - passed
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.integrity_check --json`
  - status `ok`, warnings only for acknowledged noncanonical brain sources

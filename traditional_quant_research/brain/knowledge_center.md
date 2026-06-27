# Traditional Quant Research 知识对象

## Object Classes
### class `baostock_only_personal_quant`
`definition`: personal capital oriented, Baostock-available data first, candidate selection through formal personal gate.
`endpoint`: `personal_backtest_candidate`, with post-selection recommendation `user_discretion`.
`not`: institution-grade promotion, paper/live plan or production execution.

### class `formal_personal_gate`
`inputs`: 2017-2026 profile, fees, impact, execution constraints, weak-year gate, return gate, drawdown, exposure/fallback evidence.
`outputs`: `personal_backtest_candidate`, `personal_research/backtest_only`, or `diagnostic`.
`invariant`: relaxed smoke, single-year evidence or incomplete walk-forward remains diagnostic.

### class `true_size_gate`
`accepted_sources`: authenticated PIT `tushare.daily_basic` or equivalent authenticated PIT size source.
`diagnostic_sources`: CNInfo/AkShare/efinance/current quote/proxy amount and Baostock amount/turnover proxies.
`invariant`: diagnostic size proxies do not set true-size `daily_size_ready_for_research=True`.

### class `weak_year_rebuild`
`fixed_weak_years`: `2017/2018/2022/2023`
`semantics`: fit rules use years before eval year; eval-year oracle fields describe potential, not selectable rules.

### class `ml_signal_rebuild`
`modes`: `baseline`, `weak_weighted`, `regime_weighted`, `regime_heads`.
`invariant`: predictions retain unified `score` plus training_mode/model_head/regime/sample_weight audit; fit does not use eval year data.

## Long-Term Lessons
- Traditional factors and ML signals need cost-aware, OOS, weak-year and exposure evidence together.
- High mean annualized return cannot replace weak-year gate.
- Capital scaling can improve risk budget for weak regimes, but alpha still needs independent strength.
- Factor family expansion, pruning, generic regime fallback and LightGBM are useful research tools, not automatic candidate promotion.
- Shortline limit-up style research must separate "strong" from "buyable"; executable-only evidence controls candidate status.
- Combination/exposure constraints improve evaluation only when fallback and active exposure evidence are explicit.

## Current Method Index
- `frontier_personal_candidate_selection_report`: current candidate selection summary entry.
- `frontier_ml_signal_rebuild`: ML signal rebuild and training-mode audit.
- `frontier_weak_year_rebuild`: prior-fit weak-year/regime diagnostics.
- `frontier_personal_protocol_grid`: formal personal grid and gate input.
- `frontier_personal_candidate_gate`: candidate classification.

## Pure Functions
- `is_formal_candidate_evidence(result) -> bool`
- `is_true_size_ready(source) -> bool`
- `is_user_discretion_candidate(candidate) -> bool`
- `should_write_research_log(change) -> bool`

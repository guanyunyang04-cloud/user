# Daily Research V2 Horizon 30d Soft Penalty Execution-Candidate Review

Date: `2026-06-03`

## Summary

- Status: `v2_horizon_30d_soft_penalty_execution_candidate_review / evidence_grade_candidate_backtest / research-only / execution-frozen`.
- Research program: `daily_research_v2_research_reset`.
- Study families: `v2_score_backtest_bridge`, `v2_candidate_review_matrix`, `v2_bad_month_attribution`, `v2_selective_throttle_matrix`.
- Source anchor: `mh_v2_horizon_concentration_train_contract_anchor_20260603_01`.
- Source seed runs: `mh_v2_horizon_30d_soft_penalty_seed7_20260603_01`, `mh_v2_horizon_30d_soft_penalty_seed11_20260603_01`, `mh_v2_horizon_30d_soft_penalty_seed19_20260603_01`.
- Dataset id: `policy_input_bundle__45e3d8c059ba718426a9f887`.
- Pool id: `policy_pool_view__925e8604a91a9c07a5387fb1`.
- Feature profile: `raw_kline_context_v2_tradeable_local_state_v1`.
- Active artifact impact: `daily_research/output/active_execution_strategy.json` unchanged.

## Verdict

- The 3-seed forecast-gate-pass candidate transfers into execution-candidate review, but the raw score-to-weight bridge is not promotion-review eligible.
- The narrow candidate review matrix completed `27/27` shared backtests with `0` promotion-review eligible variants.
- Bad-month attribution shows the blocker is dominated by high-score / high-weight exposure interacting with local volatility and reversal states, not trading cost.
- Selective high-volatility throttle materially improves the execution-candidate profile but still does not close the monthly gate.
- The best repair candidate remains `research-only`: strong excess Sharpe and drawdown repair, but positive month ratio is still `7/11`, below the `>=0.75` gate.

## Bridge Evidence

- Run tag: `v2_score_backtest_bridge_horizon_30d_soft_penalty_20260603_01`.
- Source anchor: `mh_v2_horizon_concentration_train_contract_anchor_20260603_01`.
- Source seeds: `mh_v2_horizon_30d_soft_penalty_seed7_20260603_01`, `mh_v2_horizon_30d_soft_penalty_seed11_20260603_01`, `mh_v2_horizon_30d_soft_penalty_seed19_20260603_01`.
- Shared backtest shape:
  - holding count `20`;
  - max weight `0.08`;
  - rebalance `5d / all offsets`;
  - costs transaction `3 bps`, slippage `7 bps`, sell tax `10 bps`.
- Metrics:
  - annual return `0.539385`;
  - excess annual return `0.263178`;
  - excess Sharpe `0.999725`;
  - max drawdown `-0.302811`;
  - positive month ratio `0.545455`;
  - negative months `5`;
  - worst month `-0.081760`;
  - issue flags: `low_positive_month_ratio`, `deep_bad_month`, `concentrated_positive_months`.
- Compared with strict v2 bridge `v2_score_backtest_bridge_20260602_01`, the new branch slightly improves excess Sharpe but worsens monthly stability and drawdown. It was therefore allowed to enter matrix review as a diagnostic candidate, not as an execution upgrade.

## Candidate Matrix Evidence

- Run tag: `v2_candidate_review_matrix_horizon_30d_soft_penalty_20260603_01`.
- Source bridge: `v2_score_backtest_bridge_horizon_30d_soft_penalty_20260603_01`.
- Matrix scope:
  - holding count `10/20/30`;
  - max weight `0.04/0.06/0.08`;
  - rebalance `3d/5d/10d`, all offsets;
  - market regime filter `off`;
  - costs transaction `3 bps`, slippage `7 bps`, sell tax `10 bps`.
- Completed shared backtests: `27`.
- Promotion-review eligible variants: `0`.
- Best by excess Sharpe:
  - variant `h30_mw040_rb10d_all_c3_7_10_regime_off`;
  - annual return `0.641854`;
  - excess annual return `0.347262`;
  - excess Sharpe `1.627628`;
  - max drawdown `-0.271978`;
  - positive month ratio `0.545455`;
  - negative months `5`;
  - worst month `-0.071846`;
  - issue flags: `low_positive_month_ratio`, `deep_bad_month`, `concentrated_positive_months`.

## Bad-Month Attribution

Attribution source variant: `h30_mw040_rb10d_all_c3_7_10_regime_off`.

Explicit bad-month runs:

- `v2_bad_month_attribution_horizon_30d_soft_penalty_20260603_01_202408`;
- `v2_bad_month_attribution_horizon_30d_soft_penalty_20260603_01_202406`;
- `v2_bad_month_attribution_horizon_30d_soft_penalty_20260603_01_202405`.

Negative months below `-0.03` excess return:

| month | excess return | primary pattern |
| --- | ---: | --- |
| `2024-08` | `-0.071846` | high-score / high-weight names with high volatility and reversal/extreme-state buckets |
| `2024-06` | `-0.046707` | high-score / high-weight exposure with high volatility and weak reversal buckets |
| `2024-05` | `-0.044717` | high-score / high-weight exposure; one very large single-name shock plus high volatility/reversal buckets |

Blocker classification:

- `local_volatility_issue`;
- `reversal_issue`;
- secondary `mapping_issue` because the top score bucket repeatedly receives large target weights in adverse local states.

Not primary blockers:

- `cost_turnover_issue`: monthly trading cost in bad months is too small to explain the loss.
- `market_state_issue`: the loss concentration appears at score / symbol state level, not only broad benchmark movement.
- `rebalance_timing_issue`: not proven by this evidence; it remains a follow-up hypothesis only.

## Repair Evidence

- Run tag: `v2_selective_throttle_matrix_horizon_30d_soft_penalty_20260603_01`.
- Source bridge: `v2_score_backtest_bridge_horizon_30d_soft_penalty_20260603_01`.
- Matrix scope:
  - holding count `20`;
  - max weight `0.06/0.08`;
  - rebalance `5d`, all offsets;
  - throttle modes `high_volatility`, `recent_runup_or_high_volatility`;
  - penalties `0.02/0.05`;
  - recent return window `5`;
  - volatility window `20`;
  - recent return quantile `0.80`;
  - volatility quantile `0.80`.
- Completed shared backtests: `8`.
- Promotion-review eligible variants: `0`.

Best repair candidate:

- variant `h20_mw060_rb5d_all_c3_7_10_thr_high_volatility_p0p05_rrw5_vw20_rq80_vq80`;
- annual return `0.867981`;
- excess annual return `0.532816`;
- excess Sharpe `2.460935`;
- max drawdown `-0.195756`;
- positive month ratio `0.636364`;
- negative months `4`;
- worst month `-0.027145`;
- annual return cost drag `0.153744`;
- issue flags: `concentrated_positive_months`;
- guarded cell ratio `0.195422`;
- original top20 retained ratio `0.594076`.

Interpretation:

- High-volatility throttle repairs the deep-bad-month and drawdown blocker.
- It does not repair the monthly positive ratio / negative month count gate.
- No local risk cap was run in this pass because the post-throttle blocker is no longer target-weight concentration or deep bad month; it is residual month-win consistency and concentrated positive-month dependence.

## External Research Comparison

Reference coordinates:

- Gu, Kelly, and Xiu, `Empirical Asset Pricing via Machine Learning`: machine-learning return prediction can produce economic gains, but needs out-of-sample economic validation rather than prediction metrics alone.
- Microsoft Qlib: a mature quant research platform separates data, model, backtest, portfolio construction, and production workflow.
- FinRL: portfolio/RL research frames continuous portfolio decisions, but production use still depends on transaction costs, risk controls, and live/paper validation.

Where this project is stronger than many notebook or paper-demo workflows:

- explicit dataset id and pool id;
- PIT / strict tradeable pool discipline;
- multi-seed forecast evidence before candidate review;
- active artifact guard;
- cost-aware shared backtests;
- candidate gate with monthly quality, drawdown, Sharpe, bad-month, and cost-drag checks;
- explicit evidence grade and brain writeback discipline.

Where this project is still below institutional production readiness:

- no live paper track record for the v2 candidate;
- no capacity / market-impact model beyond current cost assumptions;
- no broker-fill or order-book execution validation;
- no production monitoring, degradation alarms, or rollback playbook for this candidate;
- cross-regime and PIT/status hardening are still incomplete;
- execution bridge remains frozen by governance.

## Next Allowed Actions

- Keep `horizon_30d_soft_penalty_v1` as a research-only execution-candidate branch, not a promotion candidate.
- Do not run local risk cap as the immediate next step unless a new attribution pass proves target-weight concentration reappears after throttle.
- Next research step should target residual month-win consistency:
  - month-state score calibration;
  - validation-selected throttle thresholds;
  - sector/liquidity/volatility/horizon cap diagnostics;
  - post-throttle attribution for the remaining negative months `2024-05`, `2024-06`, `2024-08`, and `2024-11`;
  - avoid global state sizing unless portfolio-level path stress is proven.
- Continue comparing against strict v2 baseline and traditional-PIT baseline before any execution rebuild discussion.

## Boundaries

- `promotion_allowed=false`.
- Execution remains `frozen_skeleton_only / awaiting_research_rebuild`.
- No paper/live trading, broker integration, production-root rebuild, formal trade plan, or active artifact promotion is authorized.
- `daily_research/output/active_execution_strategy.json` remains unchanged.
- All evidence here is candidate-backtest evidence, not production target-weight evidence.

## Verification

- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/path_policy/tests/test_v2_score_backtest_bridge.py daily_research/path_policy/tests/test_v2_candidate_review_matrix.py daily_research/path_policy/tests/test_v2_bad_month_attribution.py daily_research/path_policy/tests/test_v2_selective_throttle_matrix.py -q`: `15 passed`.
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.v2_score_backtest_bridge ... --run-tag v2_score_backtest_bridge_horizon_30d_soft_penalty_20260603_01 --run-backtest --json`: completed.
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.v2_candidate_review_matrix ... --run-tag v2_candidate_review_matrix_horizon_30d_soft_penalty_20260603_01 --run-backtests --json`: completed, `27` shared backtests, `0` promotion-review eligible.
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.v2_bad_month_attribution ... --month 2024-08/2024-06/2024-05 --json`: completed.
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.v2_selective_throttle_matrix ... --run-tag v2_selective_throttle_matrix_horizon_30d_soft_penalty_20260603_01 --run-backtests --json`: completed, `8` shared backtests, `0` promotion-review eligible.
- `git diff -- daily_research/output/active_execution_strategy.json`: empty after runs.

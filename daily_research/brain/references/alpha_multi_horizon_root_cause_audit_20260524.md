# Alpha Multi-Horizon Root-Cause Audit - 2026-05-24

## Scope
- Status: `research / root-cause audit / shadow-only / no training launched by audit`.
- Mainline: `alpha_multi_horizon_utility_policy_v1`.
- Source study tag: `path20_horizon_discovery_no_alpha_gru_liquid500_h1_2_3_5_8_10_15_20_30_du_cost20_hit10_dd010_20260523_01`.
- Audit artifact tag: `mh_utility_horizon_root_cause_audit_20260524_01`.
- Dataset id: `policy_input_bundle__7c8f58d851bce8179e1e9e2d`.
- Feature profile: `raw_kline_context_no_alpha_prior_v1`.
- Model family: `gru_sequence_static_context`; seed `7`.
- Role years: train `2019-2022`, validation `2023`, test `2024`.
- Active artifact impact: `daily_research/output/active_execution_strategy.json` unchanged.

## Artifacts
- `daily_research/output/path_policy/studies/mh_utility_horizon_root_cause_audit_20260524_01/horizon_root_cause_audit.json`
- `daily_research/output/path_policy/studies/mh_utility_horizon_root_cause_audit_20260524_01/horizon_root_cause_audit.csv`
- `daily_research/output/path_policy/studies/mh_utility_horizon_root_cause_audit_20260524_01/horizon_root_cause_audit.md`

## Verdict
- Fact: the read-only audit completed with conclusion enums:
  - `true_long_horizon_edge`
  - `target_bias_to_long`
  - `horizon_head_collapse`
  - `short_feature_insufficient`
  - `training_instability`
- Inference: the current result should not be simplified to "this model is naturally only long-term"; the evidence says the present input/GRU/decision-utility setup finds stronger 20-30d signal while the horizon head collapses too aggressively to 30d.
- Inference: short horizon is not dead, but current daily input and objective are not enough to prove a robust 1-5d tradable edge.
- Boundary: this audit does not authorize production retrain, live/default, active manifest edits, production root edits, allocator/replay, trade-plan bridge, paper account bridge, or broker actions.

## Key Evidence
- Validation per-horizon rank IC / spread:
  - `1d`: rank IC `-0.031868`, spread `0.000522`, monthly positive `63.6%`.
  - `5d`: rank IC `0.058565`, spread `0.008574`, monthly positive `90.9%`.
  - `10d`: rank IC `0.082812`, spread `0.016576`, monthly positive `81.8%`.
  - `20d`: rank IC `0.113206`, spread `0.033301`, monthly positive `90.9%`.
  - `30d`: rank IC `0.131024`, spread `0.047129`, monthly positive `90.9%`.
- Test per-horizon rank IC / spread:
  - `1d`: rank IC `-0.029043`, spread `-0.000535`, monthly positive `54.5%`.
  - `5d`: rank IC `0.030911`, spread `0.007263`, monthly positive `81.8%`.
  - `10d`: rank IC `0.045247`, spread `0.013858`, monthly positive `81.8%`.
  - `20d`: rank IC `0.064482`, spread `0.028690`, monthly positive `63.6%`.
  - `30d`: rank IC `0.087132`, spread `0.045881`, monthly positive `72.7%`.
- Predicted best horizon collapse:
  - validation `pred_30d_share = 83.7%`, `future_long_share = 44.7%`.
  - test `pred_30d_share = 95.7%`, `future_long_share = 43.6%`.
- Training audit:
  - seed `7`, `epochs_ran = 8`, `best_epoch = 2`, `stopped_reason = early_stopping_patience_exhausted`.
  - audit warning: `early_best_epoch_warning`.
- Target audit:
  - validation hit base rate `0.812857`; test hit base rate `0.811302`.
  - utility decile monotonicity passed on validation and test.
- Post-hoc calibration context:
  - calibration gate remained `failed_regression`.
  - `top2_blend` was a near miss but did not pass the validation spread floor.

## Interpretation
- Fact: longer horizons have stronger rank/spread in both validation and test, so there is real evidence for a longer-horizon utility edge in this setup.
- Fact: `future_best_horizon` remains distributed; it does not collapse to 30d like `pred_best_horizon`.
- Inference: part of the observed long tilt is likely real signal strength, while part is model/head/objective behavior.
- Inference: the strongest immediate blocker is not "long horizon exists"; it is that the model does not yet behave like a reliable per-sample horizon chooser.
- Inference: short-line profitability cannot be judged by hit rate alone; it needs prediction signal, cost-adjusted execution, turnover, slippage sensitivity, and drawdown stability.

## Next Allowed Actions
- First run shadow-only target-function controls with the same dataset, pool, split, model family, and seed:
  - `mh_short_utility_1_3_5d_v1`
  - `mh_mid_utility_5_10_20d_v1`
  - `mh_long_utility_15_20_30d_v1`
- Only if a target family passes validation/test prediction gates should architecture controls run:
  - `gru_sequence_static_context`
  - `patch_transformer`
  - `stock_mixer_sequence`
- Only if prediction gates pass should execution simulation compare against:
  - `short_expert_policy_v5b`
  - `short_expert_monthly_v1`
- A single seed can only support `continue research` or `stop research`; it cannot support production claims.

## Boundaries
- No live/default.
- No active manifest mutation.
- No production root mutation.
- No production retrain or promotion.
- No allocator/replay/trade-plan/paper-account bridge.
- No broker order path.

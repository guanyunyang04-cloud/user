# r67 Paper-DFL Replace Portfolio-Set V5 Status

日期：`2026-05-14`

## Summary
- Status: `research / shadow-only / paper-mechanism reproduction`.
- Production anchor: `daily_research/output/active_execution_strategy.json` unchanged.
- Live default remains `short_expert_policy_v5b__regoff_k1_20d_ensemble_native_anchor__active`.
- r67 directly replaces the internal `portfolio_set_v5` training target, loss, prediction translation, and active r65 profile default with `portfolio_set_v5_dfl_pg_v1`, while preserving backend name `formal_torch_portfolio_set_v5`, artifact type, loader, and predict API.
- First-stage evidence passed for code contracts, synthetic oracle/PG behavior, and a tiny strict-Gold direct protocol smoke; it is not a promotion verdict and not a strategy-success verdict.

## Implementation Facts
- File changed: `daily_research/continuous_policy/model_portfolio_set_v5.py`.
- New internal version: `PORTFOLIO_SET_V5_INTERNAL_VERSION = "portfolio_set_v5_dfl_pg_v1"`.
- Old v5 artifact loading remains compatible; new v5 training now records `loss_profile=portfolio_set_v5_dfl_pg_v1` and model config `portfolio_set_v5_internal_version=portfolio_set_v5_dfl_pg_v1`.
- Target construction no longer wraps `core_v4_release_first_targets`; it now builds a decision object with:
  - `source_supply`
  - `receiver_demand`
  - `cash_buffer`
  - `target_weight`
  - `turnover_budget`
  - `risk_budget`
  - `constraint_violation`
  - `decision_value`
- Added torch oracle `project_portfolio_set_v5_cashflow_oracle(...)`:
  - long-only
  - single-name cap `0.24`
  - source / receiver / cash conservation
  - turnover, risk, cash floor, cost, and feasibility diagnostics
- Added PG-DFL style surrogate:
  - positive / negative score perturbation around source and receiver targets
  - decision oracle value margin loss
  - oracle target loss and constraint-violation loss
- Prediction now derives release-first target fields from oracle-compatible output and keeps `target_weight` / `target_delta` direction coherent.
- Active r65 profile default in `research_profile_registry.py` changed from `alpha_result_value_budget_split_v48` to `portfolio_set_v5_dfl_pg_v1`; old v48 remains a supported compatibility alias, not the default new line.
- Behavior audit now preserves evaluation universe sizing by reading `max_universe_size` / `prepared_summary.universe_size` from evaluation summary before recomputing teacher rollout. This fixes the r67 smoke audit failure where audit ignored `--max-universe-size 80` and fetched full TDX universe.

## Test Evidence
- Focused v5/audit/profile contracts:
  - `test_behavior_audit_prepare_contract.py`
  - `test_portfolio_set_v5_release_first_loss.py`
  - `test_portfolio_set_v5_day_set_dataset.py`
  - `test_portfolio_set_v5_artifact_contract.py`
  - `test_portfolio_set_v5_training_contract.py`
  - `test_protocol_profile_binding.py`
  - `test_research_registry_simplification.py`
- Full continuous-policy suite:
  - Command: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/continuous_policy/tests -q`
  - Result: `210 passed, 24 warnings`
  - Warnings are existing cvxpylayers / numpy deprecation warnings, not r67 failures.
- Guards:
  - `git diff --check`: passed; warning only that `research_profile_registry.py` CRLF will be normalized to LF when Git touches it.
  - `C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/tools/doc_guard.py check`: passed.
  - `C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/tools/brain_integrity_check.py --json`: `status=ok`.
  - `git diff -- daily_research/output/active_execution_strategy.json`: empty.

## Protocol Evidence
- Failed / partial runs:
  - `protocol_r67_paper_dfl_replace_v5_smoke_20260514_01`: failed during train with CUDA OOM; diagnostic only.
  - `protocol_r67_paper_dfl_replace_v5_smoke_20260514_02`: train/evaluate/shadow/export wrote a protocol summary, but behavior audit failed because audit ignored smoke universe sizing; partial evidence only.
  - `_02` audit was later rechecked with tag `protocol_r67_paper_dfl_replace_v5_smoke_20260514_02__audit_recheck_after_max_universe_fix` and passed; this validates the audit fix but does not retroactively convert `_02` into a completed direct protocol verdict.
- Completed smoke:
  - Protocol tag: `protocol_r67_paper_dfl_replace_v5_smoke_20260514_04`.
  - Protocol summary: `daily_research/output/continuous_policy/protocols/protocol_r67_paper_dfl_replace_v5_smoke_20260514_04/protocol_summary.json`.
  - Dataset: `continuous_policy_training_matrices__strict_train__36c234208d5f375ea1cccfc1`.
  - Backend: `formal_torch_portfolio_set_v5`.
  - Loss profile: `portfolio_set_v5_dfl_pg_v1`.
  - Internal version: `portfolio_set_v5_dfl_pg_v1`.
  - Completed stages: train, evaluate, shadow, export, behavior-audit, conclusion-ledger.
  - Training evidence: `insufficient`.
  - Failed training checks: `teacher_action_rows`, `best_epoch_not_at_edge`.
  - Promotion gate: `shadow_only`.

## Key Metrics From `_04`
- Training / paper reproduction:
  - `paper_reproduction_pg_surrogate_loss=73.68511962890625`.
  - `decision_oracle_constraint_violation_mean=0.0`.
  - `decision_oracle_source_l1=0.0003794828080572188`.
  - `decision_oracle_receiver_l1=0.00024994355044327676`.
  - `decision_oracle_target_weight_l1=0.0005965111195109785`.
  - `target_source_count=2674.0`.
  - `target_receiver_count=55.0`.
  - `target_intent_translation_conflict_count=0.0`.
- Evaluation:
  - `portfolio_daily_receiver_target_count=320.0`.
  - `portfolio_daily_source_target_count=0.0`.
  - `intent_translation_conflict_rate=0.0`.
- Shadow:
  - `portfolio_daily_receiver_target_count=328.0`.
  - `portfolio_daily_source_target_count=0.14285714285714285`.
  - `intent_translation_conflict_rate=0.7415795586527294`.
- Behavior audit diagnosis:
  - `release_translation_deploy_failure_mode=funding_release_not_observed`.
  - Deploy is realized, but explicit source funding / release behavior is still not closed.

## Facts
- r67 paper-driven rewrite is implemented in the v5 kernel and active r65 profile default.
- The synthetic PG/oracle fixture and full continuous-policy unit contracts pass.
- A tiny strict-Gold direct protocol smoke with default r65 profile now completes all protocol stages and records DFL-PG v1 metadata.
- Active execution artifact is unchanged.
- r67 does not promote continuous_policy and does not alter live/default.

## Inferences
- r65's original source/receiver failure was not only model capacity; target construction, decision objective, and prediction-to-simulator translation needed a shared cashflow semantics object.
- The r67 DFL-PG v1 rewrite fixes the first engineering layer: target fields, oracle diagnostics, loss terms, artifact metadata, and release-first output coherence.
- Behavior is still not fully closed: evaluation source remains zero and shadow intent translation conflict remains high, so live or promotion discussion is still invalid.

## Assumptions
- Strict Gold dataset `continuous_policy_training_matrices__strict_train__36c234208d5f375ea1cccfc1` remains the training-safe source for the next v5 research iteration.
- The tiny smoke window is sufficient for mechanism and wiring evidence, but insufficient for final strategy judgment.
- GPU availability allowed smoke execution; CPU-only environments should first run unit/synthetic reproduction and postpone wider strict-Gold smoke without lowering evidence standards.

## Boundaries
- Do not treat `_01`, `_02`, or `_02` audit recheck as completed protocol verdicts.
- Do not treat `_04` as promotion, confirmatory, formal evidence, or strategy-success evidence.
- Do not modify `daily_research/output/active_execution_strategy.json`.
- Do not discuss live/default change from this evidence.
- Do not use loose `latest_*` as truth; use explicit tag `protocol_r67_paper_dfl_replace_v5_smoke_20260514_04`.

## Next Allowed Actions
- Continue research/shadow-only on `portfolio_set_v5_dfl_pg_v1`.
- Fix prediction-to-simulator behavior so strict-Gold evaluation produces nonzero, explainable source targets, not only receiver targets.
- Investigate shadow `intent_translation_conflict_rate=0.7415795586527294` before any broader study.
- After source/translation closure improves, run a longer strict resume or wider strict-Gold smoke; training evidence must become sufficient before stronger conclusions.
- Keep r61/r65 old evidence as baseline/ablation only; do not revive legacy direct-protocol profiles as active entrypoints.

# Daily Research Current Frontier Compaction 20260603

## Verdict
- Status: `brain_maintenance / current_frontier_compaction / research-only / shadow-only`.
- Scope: `daily_research` brain hot-path maintenance, frontier freshness reconciliation, and guard readability.
- Active artifact impact: `daily_research/output/active_execution_strategy.json` remains unchanged and was not restored, regenerated, or promoted.
- This reference preserves details removed from `daily_research/brain/state_center.md` and `daily_research/brain/operations_center.md` during hot-path compaction.

## Current State
- `daily_research_v2_research_reset` remains the current path_policy / multi-horizon research mainline.
- Strict v2 baseline `mh_v2_reset_tradeable_mainboard_anchor_20260601_01` remains the pass-grade research baseline, not promotion-grade.
- Traditional-PIT comparison `mh_v2_traditional_pit_tradeable_mainboard_anchor_20260602_01` remains stricter-data-contract comparison evidence with `near_pass`, not replacement authority.
- Score bridge, candidate review, bad-month attribution, risk overlay, selective throttle, state sizing, local-state input/loss, horizon concentration repair, and horizon train-contract runs are research-only evidence.
- `horizon_30d_soft_penalty_v1` has three seed forecast-gate pass evidence, but the next required step is research-only score-backtest bridge and comparison against strict v2 baseline, traditional-PIT baseline, and local-state input/loss branches.

## Frontier Reconciliation
- Run tags:
  - `mh_v2_local_state_loss_calibration_score_monthly_robust_v1_seed7_20260603_01`
  - `mh_v2_local_state_loss_calibration_risk_drawdown_reweighted_v1_seed7_20260603_01`
- Both run tags exist under `daily_research/output/path_policy/studies/` with `study_summary.json`.
- Both summaries report `status=completed`, `stage=forecast_walkforward_study`, `evidence_verdict=forecast_test_confirmed`, `shadow_only=true`, `promotion_allowed=false`, and `active_execution_strategy_expected_diff=none`.
- Both runs use lake dataset `policy_input_bundle__45e3d8c059ba718426a9f887`, strict tradeable mainboard pool `policy_pool_view__925e8604a91a9c07a5387fb1`, and feature profile `raw_kline_context_v2_tradeable_local_state_v1`.
- Interpretation: these seed7 loss-calibration diagnostics are valid completed forecast evidence, but they are not v2 baseline replacement, not candidate backtest evidence, not promotion-grade, and not execution unfreeze authority.

## Compacted State Details
- The 2026-05-31 `output/cache` deletion remains a file-backed replay and active inspection blocker, not a reason to invalidate user-confirmed brain conclusions.
- The 2026-06-01 research-only rebuild established BaoStock-first new-lineage substrate but did not restore old Stage 2.8 / Stage 3G / short_v5b payloads.
- The 2026-06-01 mainboard correction excludes prefixes `300,301,688,689`; corrected mainboard pool evidence is in `alpha_multi_horizon_mainboard_pool_correction_20260601.md`.
- The 2026-06-02 v2 dataset contract upgrade produced the current pass-grade strict baseline; later risk/candidate work remains research-only.
- Continuous policy recent review sequence remains shadow-only; translation closure and oracle feasibility improved, but training evidence and behavior quality are still blockers.

## Compacted Operation Details
- Long command families for rebuild lineage, mainboard pool construction, TQ/BaoStock audit, V2 strict baseline, traditional PIT bridge, score bridge, candidate matrix, risk overlay, selective throttle, state sizing, local-state input/loss, and horizon repair should be retrieved through their dated references and `evidence_registry.json`.
- The hot path should keep only stable command entrypoints: capsule, current-frontier, evidence-index, query, verify-plan, doc_guard, integrity_check, and the active artifact diff guard.
- Long tasks must remain supervised through PID, persistent stdout/stderr, progress files, run tags, and expected artifacts; `Wait-Process -Id <pid> -Timeout 7200` is a per-window wait, not a failure threshold.

## Next Allowed Actions
- Rebuild `daily_research/brain/references/evidence_registry.json` after this reference is added.
- Run `current-frontier --json`; expected outcome is no unregistered latest run tags for the two reconciled seed7 diagnostics.
- Continue v2 research by bridging `horizon_30d_soft_penalty_v1` into research-only score backtest and comparison, without touching live/default/promotion.
- Keep active artifact missing-state handling explicit and readable in guard output; do not reconstruct active artifact from brain text.

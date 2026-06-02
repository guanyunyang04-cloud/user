# Daily Research V2 Research Framework Contract - 2026-06-02

## Summary

- Status: `daily_research_v2_research_framework_contract_accepted / research-only / execution-frozen`.
- Research program: `daily_research_v2_research_reset`.
- User decision: the v2 research framework is the full evidence chain from data contracts to execution-candidate contracts, not only model training.
- Current pass-grade research baseline: `mh_v2_reset_tradeable_mainboard_anchor_20260601_01`.
- Current stricter data-contract comparison baseline: `mh_v2_traditional_pit_tradeable_mainboard_anchor_20260602_01`.
- Active artifact impact: `daily_research/output/active_execution_strategy.json` unchanged.

## Framework Shape

The accepted v2 framework is:

```text
research objective
-> data and dataset contract
-> pool and PIT status contract
-> sample, label, and target contract
-> feature and model-input contract
-> model architecture and implementation contract
-> model output and portfolio-semantics contract
-> loss and training objective contract
-> evaluation, backtest, and gate contract
-> execution-candidate contract
-> evidence governance and writeback contract
```

This framework is the default lens for future `daily_research_v2_research_reset` work.

## 1. Research Objective Contract

- The v2 objective is not strict replay of old Stage 2.8 / Stage 3G / `short_v5b`.
- The v2 objective is to rebuild a traceable mainboard, tradeable, multi-horizon trading-utility research system on the BaoStock-first lake.
- Current score target is trading-useful cross-sectional ranking / utility, not raw price prediction alone.
- Current core metrics are rank IC, top-bottom spread, hit lift, monthly stability, negative month count, and 30d concentration.
- Old `156` feature schema and `short_v5b` payloads remain historical priors or side bridges only; they do not block v2 research.

## 2. Data And Dataset Contract

- Formal research must use explicit dataset ids, not loose `latest_*`.
- Current source market dataset: `policy_input_bundle__45e3d8c059ba718426a9f887`.
- The lake is the research storage truth; online providers may refresh/import data only through data-platform boundaries.
- Dataset quality must keep coverage, benchmark open/close, missing/fill, Amount unit, source lineage, and manifest checks visible.
- Current v2 dataset contract audit says usable but still phase-2 hardening remains around PIT history and missing/fill semantics.

## 3. Pool And PIT Status Contract

- The v2 default universe is mainboard-only and tradeable.
- Exclude ChiNext and STAR prefixes: `300,301,688,689`.
- PIT/status fields must distinguish listed, mainboard, common A share, ST, suspended, delisted, tradeable, has bar, and reject reason.
- Current pass-grade v2 strict pool: `policy_pool_view__925e8604a91a9c07a5387fb1`.
- Current stricter traditional-PIT comparison pool: `policy_pool_view__0af1d96b413fb8927270b680`.
- A stricter pool is not automatically better for model quality; it must pass the same model gate before replacing the baseline.

## 4. Sample, Label, And Target Contract

- Sample construction must explicitly record train / validation / test split, role counts, feature store shape, source pool id, and label semantics.
- Current label family is `next_open_entry_to_future_open`.
- Current execution assumption for labels is next-open entry.
- Horizons are multi-horizon, currently `1,2,3,5,8,10,15,20,30`.
- Cost, hit threshold, drawdown penalty, future decision score, future best horizon, and hit labels are part of the target contract.
- TQ/BaoStock lineage audit can prove data/label equivalence; it is not a model-quality verdict by itself.

## 5. Feature And Model-Input Contract

- Model input is a first-class research object, not an incidental preprocessing step.
- Current v2 feature profile: `raw_kline_context_v2_tradeable_amount_checked`.
- Current pass baseline feature count: `124`.
- Feature groups include state, raw kline, market context, peer context, regime context, and history quality.
- Alpha/score-like leakage features must remain explicitly audited; current v2 baseline has alpha-like feature count `0`.
- Amount-sensitive features must record `amount_unit_policy`, `amount_unit_factor`, and consistency diagnostics.
- Feature/input changes must be compared against the current v2 strict baseline and, where useful, the traditional-PIT strict comparison baseline.

## 6. Model Architecture And Implementation Contract

- Architecture is important but downstream of data, target, and input contracts.
- Current pass-grade model family: `gru_sequence_static_context`.
- Alternative architectures are allowed only as explicit experiments with declared evidence grade and comparable dataset/pool/feature/loss bindings.
- Implementation artifacts must record seed, model family, feature profile, horizon grid, loss, output profile, training budget, early stopping, manifest, and paths.
- Low-budget architecture scouts cannot replace the baseline or trigger execution work.

## 7. Model Output And Portfolio-Semantics Contract

- Model output must define its operational meaning.
- Current main research score is `pred_decision_score`.
- Output may represent ranking score, utility score, horizon forecast, best-horizon choice, or target-weight precursor; these meanings must not be conflated.
- A score panel is not yet a production target-weight panel.
- Any future output-to-weight mapping must define rebalance cadence, caps, turnover, cost, constraints, and execution assumptions before it can become an execution candidate.

## 8. Loss And Training Objective Contract

- Loss is how the trading objective is encoded into learning; it cannot be judged apart from labels and outputs.
- Current pass-grade loss: `target_norm_head_constraint_v1`.
- New losses must declare what behavior they optimize: rank, hit, utility, horizon calibration, concentration control, drawdown sensitivity, or portfolio semantics.
- Training results must separate run completion from evidence quality.
- Completed runs with insufficient seed count, insufficient epoch budget, best epoch at edge, or missing convergence evidence remain smoke/scout evidence.

## 9. Evaluation, Backtest, And Gate Contract

- Current v2 evidence-grade gate requires:
  - 3 seeds;
  - test rank IC all positive;
  - top-bottom spread all positive;
  - hit lift all positive;
  - mean monthly positive rate `>=0.75`;
  - max negative months `<=2`;
  - 30d concentration not worse than reference threshold.
- Current pass-grade baseline: `mh_v2_reset_tradeable_mainboard_anchor_20260601_01`.
- Current traditional-PIT strict comparison anchor is `near_pass`, not replacement baseline, because `negative_month_count_max=3`.
- Future backtests must unify dataset, pool, costs, rebalance rule, benchmark, split, score semantics, target-weight semantics, and next-open assumptions.

## 10. Execution-Candidate Contract

- Execution remains frozen: `frozen_skeleton_only / awaiting_research_rebuild`.
- Research evidence does not automatically authorize live/default, paper/broker, trade-plan production, production-root rebuild, or active manifest promotion.
- Execution rebuild requires a separately approved plan after research evidence closes:
  - candidate score-to-weight mapping;
  - cost / slippage / rebalance policy;
  - risk and position constraints;
  - data readiness and manual runbook;
  - replay / paper / broker adapter design;
  - explicit active-artifact boundary.

## 11. Evidence Governance Contract

- Every material experiment must state evidence grade: `smoke_only`, `scout_only`, `evidence_grade`, or `promotion_grade`.
- Every major conclusion must separate facts, inferences, assumptions, and boundaries.
- Registry fields remain:
  - `research_programs` for stable mainlines;
  - `study_families` for stage / experiment families;
  - `run_tags` for physical runs.
- Explicit dataset id, pool id, feature profile, model family, loss, output profile, seed set, cost, horizon grid, and run tag are required for model-quality comparisons.
- Brain writeback goes to dated references for long evidence and to `state_center.md` / `knowledge_center.md` only as compact current pointers.

## Current Accepted Baselines

- Pass-grade v2 baseline:
  - anchor: `mh_v2_reset_tradeable_mainboard_anchor_20260601_01`;
  - dataset: `policy_input_bundle__45e3d8c059ba718426a9f887`;
  - pool: `policy_pool_view__925e8604a91a9c07a5387fb1`;
  - feature profile: `raw_kline_context_v2_tradeable_amount_checked`;
  - model/loss: `gru_sequence_static_context` + `target_norm_head_constraint_v1`;
  - gate: `pass`.
- Stricter data-contract comparison baseline:
  - anchor: `mh_v2_traditional_pit_tradeable_mainboard_anchor_20260602_01`;
  - status sidecar: `data_platform_v2_status_sidecar__37dba59cdced261cddfedf11`;
  - pool: `policy_pool_view__0af1d96b413fb8927270b680`;
  - gate: `near_pass`.

## Next Allowed Actions

- Treat this framework as the default decomposition for all future v2 research plans.
- Continue phase-2 data-contract hardening: PIT listing/delist, status source depth, missing/fill semantics, limit/industry/valuation/market-cap domains.
- Run feature/model improvements as narrow matrices against the pass-grade v2 baseline and optionally the traditional-PIT strict comparison baseline.
- Keep execution frozen until an explicitly approved execution-candidate bridge and rebuild plan exists.

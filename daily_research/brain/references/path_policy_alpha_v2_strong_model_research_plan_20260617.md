# Path Policy Alpha V2 Strong Model Research Plan 20260617

## Verdict
- Status: `research_plan_accepted / brain_pointer / research_only / execution_frozen`.
- Date: `2026-06-17`.
- Research program: `daily_research_v2_research_reset`.
- Study family: `qdp_alpha_v2_strong_model_research`.
- Current data substrate: QDP `style_structural_alpha_v2_label_v2` training pack, feature profile `style_structural_alpha_v2`, feature count `307`, label schema `path20_basic_v2` version `2`, static context `symbol/exchange/industry`.
- Current anchor run: `qdp_alpha_v2_hybrid_topn_h256_t4_b512_full_e12_20260616_01`.
- Active artifact impact: none. `daily_research/output/active_execution_strategy.json` must remain absent or unchanged unless future explicit recovery or promotion authority is given.
- Decision: subsequent `daily_research` path_policy work should first follow this plan. Multi-seed, score-backtest bridge, candidate matrix, and execution-candidate review are intentionally deferred until a stronger single-seed alpha_v2 model is found.

## Current Facts
- The latest consumable daily research pack is `mainboard_style_structural_alpha_v2_label_v2_pack_full_2012_2025_20260616_01` under QDP.
- The pack has already passed daily protocol consumption smoke with role split `64/64/64`; that smoke only proves data/model wiring.
- The first same-mouth alpha_v2 strong-model run completed with `hybrid_expert_fusion_static_context`, h256, transformer layers `4`, heads `8`, GRU layers `2`, batch `512`, seed `7`, output profile `decision_utility_v1`, loss profile `topn_excess_rank_v1`.
- Training system selected epoch `2`. Validation/test rank_ic20 are `0.076656 / 0.115449`; validation/test spread20 are `0.018850 / 0.023697`.
- `personal_topk_v1` selected `pred_decision_score top1 h10` on validation. Validation personal score is `276.55`; validation net is `2.72%`; same candidate test net is `2.73%`.
- This is better than the old v1 default h128/h192/h256 same-candidate top-K comparisons, but it is still single-seed supervised forecast evidence, not multi-seed model proof and not execution evidence.

## User Objective
- Personal small-capital quant research.
- Primary preference: efficient, high-return, top-K stock selection.
- Current priority: find a stronger alpha model first.
- Current non-priority: do not spend the next cycle on multi-seed, score-backtest bridge, candidate matrix, paper/live, broker, active/default, or promotion.
- Practical interpretation: keep the main effort on alpha_v2 hybrid model quality, output/loss alignment, and model ablation. Treat execution and stability gates as later confirmation layers.

## Non-Goals
- Do not touch `daily_research/output/active_execution_strategy.json`.
- Do not claim promotion, production, live/default, paper, broker, or execution-candidate status.
- Do not treat smoke, throughput, datecap, or single-seed runs as final model-quality evidence.
- Do not force the hybrid model to become a full same-day cross-sectional attention model in this phase.
- Do not start regime full training until the date-batch loader or cross-section pack path is verified fast enough.
- Do not tune score-backtest mappings to rescue a weak alpha model.

## Working Interpretation
- `hybrid_expert_fusion_static_context` is currently best treated as a stock-sequence alpha forecaster: each stock consumes its own 252-day dynamic window plus static context and precomputed cross-sectional features.
- It has useful cross-sectional information through input features and labels, but it is not a full date-level all-stock comparer.
- Its loss can include ranking/topN surrogates, but unless the batch is date-major, batch-level topN does not exactly mean "same-day whole-market top-K".
- That mismatch is acceptable for the current hybrid strengthening phase if evaluation remains `personal_topk_v1` plus same-candidate test confirmation.
- True same-day full-market comparison belongs to a later date-batch/cross-sectional reranker or regime layer, preferably after hybrid produces strong candidate scores.

## Primary Metrics
- Primary selection metric: `personal_topk_v1` validation selection with the same selected score/topK/horizon reported on test.
- Primary economic signals: validation/test net mean, hit rate, positive-month rate, bad-month penalty, worst month, and stability penalties from `personal_topk_v1`.
- Secondary forecast signals: rank IC by horizon, spread by horizon, upside IC, decision-score IC, and learning curve shape.
- Operational signals: training samples/sec, validation time, prediction generation time, GPU memory, checkpoint count, and whether best epoch sits at the final epoch.
- Required comparison style: compare against the alpha_v2 h256 anchor and old v1 default rows using the same validation-selected same-candidate test rule.

## Stage 0: Baseline Freeze And Tooling Audit
Purpose: make the current anchor reproducible and avoid changing two things at once.

Tasks:
- Keep `qdp_alpha_v2_hybrid_topn_h256_t4_b512_full_e12_20260616_01` as the current alpha_v2 h256 seed7 anchor.
- Preserve the command contract: QDP training pack manifest, train years `2012-2023`, validation `2024`, test `2025`, next-open labels, lookback `252`, static context enabled.
- Confirm that `personal_topk_v1`, `forecast_checkpoint_topk_reselection.py`, and same-candidate v1/v2 comparison remain available after any code change.
- Keep the accelerated test changes already present in `daily_research/path_policy/tests/fixtures.py` and `daily_research/path_policy/tests/test_forecast_training.py`.

Verification:
- Run focused tests for changed path_policy code.
- Run `git diff --check`.
- For brain writes, run changed-scope doc guard and integrity check.

Done condition:
- A future agent can reproduce the current anchor facts and knows it is the baseline to beat.

## Stage 1: Loss And Output Alignment
Purpose: make `pred_decision_score` and the training objective better match personal top-K selection without overfitting execution.

Order:
1. Audit current `topn_excess_rank_v1` semantics in `forecast_training.py` and related loss contract helpers.
2. Run or implement only one loss change at a time.
3. Keep model architecture h256/T4/GRU2/b512/seed7 fixed during first loss comparisons.
4. Always emit validation and test predictions, run `personal_topk_v1`, and compare same-candidate test results.

First candidates:
- Existing control: `topn_excess_rank_v1` on alpha_v2 h256 seed7, current anchor.
- Existing candidate if contract-compatible: `horizon_30d_soft_penalty_v1`, mainly to test whether older horizon calibration helps or hurts alpha_v2 top-K.
- Existing candidate if contract-compatible: `score_monthly_robust_v1`, mainly to test whether month robustness improves personal top-K without killing rank/spread.
- New candidate only if existing profiles do not address the gap: `decision_score_topk_alignment_v1`, a narrow auxiliary loss that aligns `pred_decision_score` with top-K useful targets while preserving current multi-horizon output contract.

Design constraints for any new loss:
- It must consume existing `path20_basic_v2` labels and prediction tensors.
- It must not require future data beyond the label horizon.
- It must preserve `decision_utility_v1` output compatibility unless a new output profile is explicitly introduced.
- It must report component losses separately so weak improvements are diagnosable.
- It must have smoke tests for tensor shapes, finite loss, and output-profile compatibility.

Graduation rule:
- A loss candidate deserves a full alpha_v2 seed7 run only after a small smoke confirms data, label, output, and backward pass.
- A loss candidate beats the anchor only if validation `personal_topk_v1` improves and same-candidate test net does not regress materially, while rank/spread and month stability do not collapse.

## Stage 2: Hybrid Architecture Ablation
Purpose: learn which parts of the hybrid model actually matter before making the model bigger.

Default fixed point:
- Hidden dim `256`.
- GRU layers `2`.
- Patch Transformer layers `4`.
- Transformer heads `8`.
- Fusion Transformer currently shallow.
- Output profile `decision_utility_v1`.

One-variable-at-a-time ablations:
- `fusion_depth`: compare current fusion depth versus a 2-layer fusion encoder if the code is extended to expose it.
- `head_depth`: compare current shallow shared head versus a deeper decision/output head if the code is extended to expose it.
- `gru_layers`: compare `2` versus `3` before increasing Transformer depth again.
- `transformer_layers`: compare `4` versus `6` only after loss alignment gives a stable direction.
- `hidden_dim`: keep `256` first; compare h192/h256/h384 only after loss/head direction is clearer.
- `expert_ablation`: no-GRU, no-Patch-Transformer, no-DLinear variants if implementation cost is low.
- `router_diagnostics`: log router weights/entropy by role/month/horizon candidate to detect expert collapse.
- `per_expert_aux`: consider auxiliary heads only after router/expert collapse is observed or ablation shows hidden unused capacity.

Reasoning:
- Current h256 is not obviously too small for stock-sequence alpha.
- Continuing to deepen only Patch Transformer is not automatically scientific; GRU depth, fusion depth, head depth, router behavior, and DLinear usefulness all need evidence.
- The first scientific step is ablation, not another blind capacity jump.

Graduation rule:
- An architecture variant must improve the loss-aligned seed7 comparison before it deserves wider seeds.
- If it is slower, it must show enough validation/test top-K gain to justify the added runtime.

## Stage 3: Expert Cooperation And Shared Channels
Purpose: test whether the three experts should communicate earlier than the current late-fusion design.

Candidate ideas:
- Shared input projection before expert-specific encoders.
- Cross-expert adapter after early encoding but before router.
- Deeper fusion token mixer over expert tokens.
- Per-expert auxiliary prediction losses to prevent one expert from becoming unused.
- Router entropy or diversity diagnostics before adding diversity regularization.

Boundary:
- Do not implement a large new architecture until Stage 1 and Stage 2 show the current bottleneck is representational rather than loss/output alignment.
- Treat any shared-channel experiment as architecture ablation, not as a proven improvement.

## Stage 4: Speed And Validation Optimization
Purpose: optimize only the true bottleneck observed during alpha_v2 runs.

Known anchor speed:
- alpha_v2 h256 seed7 full run consumed the QDP pack successfully.
- Per-epoch wall time including validation was roughly `6300-6500` seconds.
- Training throughput was roughly `925-1080` samples/sec depending on accounting.

Optimization rule:
- Measure before optimizing. Record train time, validation time, prediction time, data loading time, and diagnostics time separately where possible.
- If validation dominates, inspect validation prediction generation and metric loops before changing training.
- If data loading dominates, tune dataloader workers, pinned memory, batch size, and pack read path.
- If GPU compute dominates, compare AMP, batch size, sequence model depth, and checkpoint frequency.
- If personal top-K diagnostics dominate, optimize prediction frame reads and avoid unnecessary huge CSV rereads.

Allowed speed changes:
- Keep semantics identical unless the experiment is explicitly labeled as a new research condition.
- Prefer targeted loader/validation improvements over broad cache rewrites.
- Add tests for speed-path semantics if changing dataset or validation code.

Stop rule:
- Do not spend a full cycle optimizing speed if model-quality experiments are still cheap enough to run and the bottleneck is not blocking iteration.

## Stage 5: Candidate Graduation To Multi-Seed
Purpose: define when deferred multi-seed becomes worth doing.

Single-seed candidate can graduate only if:
- Validation `personal_topk_v1` improves over alpha_v2 h256 anchor.
- Same validation-selected candidate on test is at least comparable or better than the anchor.
- Monthly stability does not worsen in a way that explains the improvement as one-month luck.
- Rank/spread or decision-score IC does not collapse.
- Learning curve does not show obvious overfit or best-epoch-at-edge fragility.
- Runtime is acceptable for at least seeds `7,11,19`.

Then run:
- Minimum seeds: `7,11,19`.
- Preferred critical seeds: `7,11,19,23,29` only if the 3-seed result is promising and runtime is acceptable.

Boundary:
- User preference is to defer this stage for now. Do not start it until a stronger single-seed model exists or the user explicitly asks.

## Stage 6: Hybrid Plus Cross-Sectional Reranker
Purpose: use hybrid and regime/cross-sectional modeling together instead of forcing one model to solve every problem.

Recommended later design:
- Hybrid first scores the full universe as an efficient stock-sequence alpha model.
- A date-level reranker or regime model then consumes only top `300/500/1000` hybrid candidates per date.
- The reranker learns same-day relative selection among candidates, not full-universe heavy attention over all stocks unless loader throughput proves acceptable.

Prerequisites:
- Date-batch loader or cross-section training pack must be fast enough.
- A strong hybrid candidate must exist.
- Evaluation must use same-candidate validation/test top-K reporting before any score-backtest bridge.

Boundary:
- This is later work. It does not block Stage 1 and Stage 2 hybrid strengthening.

## Stage 7: Score-Backtest Bridge And Candidate Matrix
Purpose: convert strong forecast scores into research-only execution-candidate evidence.

Deferred checks:
- Monthly stability.
- Negative months and bad-month attribution.
- Drawdown.
- Turnover and transaction cost drag.
- Holding count, rebalance cadence, max weight, and concentration.
- Liquidity and limit-up/limit-down feasibility.
- Score-to-weight mapping robustness.

Boundary:
- This stage remains deferred by user preference.
- It must not be started from a weak alpha model just because bridge tooling exists.
- Passing this stage still would not automatically unfreeze live/default or active artifacts.

## Experiment Ledger Template
Each new experiment reference or summary should record:
- Run tag.
- Data manifest path and feature/label schema.
- Model family and architecture knobs.
- Output profile and loss profile.
- Seed, epochs, batch, device, AMP setting.
- Evidence grade: `smoke_only`, `scout_only`, `evidence_grade`, or `promotion_grade`.
- Validation-selected `personal_topk_v1` candidate.
- Same-candidate test result.
- Rank/spread/IC summary.
- Monthly stability and worst-month summary.
- Runtime and throughput.
- Explicit comparison against `qdp_alpha_v2_hybrid_topn_h256_t4_b512_full_e12_20260616_01`.
- Active artifact impact.

## Immediate Next Actions
1. Treat this file as the active research plan for daily_research path_policy alpha_v2 work.
2. Stage 1 first: audit and compare loss/output alignment candidates on h256 fixed architecture.
3. If a loss profile needs code changes, implement a minimal loss contract and tests before full training.
4. Run one alpha_v2 seed7 full candidate at a time, not a broad matrix.
5. After each full candidate, generate validation/test predictions, run `personal_topk_v1`, and append a compact dated reference.
6. Only after a single-seed candidate clearly beats the anchor should multi-seed be reconsidered.

## References
- `daily_research/brain/references/path_policy_qdp_pack_frontier_reconciliation_20260616.md`
- `daily_research/brain/references/path_policy_qdp_alpha_v2_h256_comparison_20260616.md`
- `daily_research/brain/references/daily_research_v2_high_return_model_discovery_v1_20260604.md`
- `daily_research/brain/references/daily_research_v2_high_return_topn_finalist_three_seed_20260606.md`
- `daily_research/path_policy/models.py`
- `daily_research/path_policy/forecast_training.py`
- `daily_research/path_policy/personal_topk_diagnostics.py`
- `daily_research/path_policy/forecast_checkpoint_topk_reselection.py`

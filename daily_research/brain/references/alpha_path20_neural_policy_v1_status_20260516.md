# alpha_path20_neural_policy_v1 Status 2026-05-16

## Verdict
- Status: `legacy / diagnostic-only / no-new-mainline-budget`.
- `alpha_path20_neural_policy_v1` is no longer the path20 main research route. It is retained only for oracle upper-bound, path-label, forecaster-baseline, and historical diagnostic evidence.
- The active path20 mainline is `alpha_path20_sequence_policy_v1`.
- The first lake tiny smoke completed with explicit study tag `alpha_path20_neural_policy_v1_tiny_smoke_20260516_01` and fixed lake dataset id `policy_input_bundle__0f116a9b78c92ff045a6853d`.
- `active_execution_strategy.json remains unchanged`; it remains guarded and must stay unchanged for this line.

## Facts
- Added path20 label generation with next-open semantics: future daily returns `1d..20d`, excess returns, q10/q50/q90 path fields, cumulative 5/10/20d labels, path drawdown/upside/risk fields, and rank/top/bottom auxiliaries.
- Added target-weight-only adapter: `portfolio_daily_target_weight` is the execution truth; source/receiver fields are derived from target delta for diagnostics and simulator compatibility only.
- Added oracle upper-bound rollout using real future path labels and existing `PortfolioState.step` allocation-layer replay.
- Added neural primitives and baselines: linear, DLinear, GRU, PatchTransformer, MLP forecasters plus pure neural target-weight allocator and path/portfolio losses.
- Legacy protocol stages remain available for diagnostics only:
  - `dataset-smoke`
  - `oracle-smoke`
  - `tiny-smoke`
- These stages now require explicit `--allow-legacy-neural-policy`. Without that flag the protocol must fail and direct the runner to `rl-*` sequence stages.
- Added focused tests under `daily_research/path_policy/tests`; command passed: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/path_policy/tests -q` with `7 passed`.

## Smoke Evidence
- Study summary: `daily_research/output/path_policy/studies/alpha_path20_neural_policy_v1_tiny_smoke_20260516_01/study_summary.json`.
- Dataset id: `alpha_path20_dataset__e96ad3bf15b4df878feb1aca`.
- Dataset smoke: 60-symbol lake slice, `2024-01-02` to `2024-02-29`, next-open labels, `1020` rows.
- Oracle upper-bound smoke: `total_return=0.17804449056406324`, `max_drawdown=-0.008167832580944112`, `return_count=16`, `native_target_valid_rate=0.9411764705882353`.
- Linear path forecaster smoke: `rank_ic_20d=-0.03937386084800192`, `top_bottom_spread_20d=-0.0015967283590900708`, `direction_accuracy_20d=0.5686274509803921`.
- Allocator smoke with oracle path labels completed and produced a positive utility proxy, but it is a tiny optimization smoke, not deployable allocator evidence.

## Inferences
- The first oracle-path smoke supports the premise that, under this simulator and short window, knowing the future path can produce positive portfolio returns.
- The negative linear forecaster smoke is expected at 2 epochs on a tiny slice; it should be treated as a baseline plumbing result, not a failure of the full direction.
- The old neural v1 route is useful as a diagnostic microscope, but it is structurally misaligned with the current pure sequence/RL goal because it centers path labels, oracle upper-bound, and forecaster/allocator decomposition.
- New path20 budget should flow to `alpha_path20_sequence_policy_v1`, where the network directly outputs target weights from historical market and portfolio state.

## Assumptions
- Lake bundle `policy_input_bundle__0f116a9b78c92ff045a6853d` remains the canonical market input source for near-term path20 smokes.
- Next-open alignment is the default execution semantics for this line unless explicitly changed.
- The target-weight adapter's derived source/receiver fields are simulator-compatibility diagnostics, not independent supervision labels.

## Boundaries
- Shadow-only: no promotion, no live/default routing, no `active_execution_strategy.json` changes.
- No loose latest: every run must cite explicit protocol/study tag and dataset id.
- Do not claim sequence/RL progress from neural v1 evidence.
- Do not use oracle upper-bound or path-label results as sequence policy success evidence.
- Do not start new mainline training from `alpha_path20_neural_policy_v1`; use it only when a diagnostic question explicitly requires the legacy protocol and pass `--allow-legacy-neural-policy`.

## Next Allowed Actions
- Run a legacy stage only for a named diagnostic purpose, for example checking oracle upper-bound replay behavior or path-label schema compatibility.
- Prefer `rl-dataset-smoke`, `rl-train-smoke`, `rl-replay-smoke`, and `rl-multiyear-smoke` for all new path20 mainline work.

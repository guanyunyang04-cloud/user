# r50 resource guard 2026-05-09

## Facts
- The r50 true solver smoke tag `self_opt_study_r50_integrated_convex_capital_flow_true_solver_smoke_20260509_1852` used the real `alpha_result_value_budget_split_v35` path with `allocation_layer_v1`.
- The run produced `checkpoint_last.pt` and `checkpoint_best.pt` under `daily_research/output/continuous_policy/models/self_opt_study_r50_integrated_convex_capital_flow_true_solver_smoke_20260509_1852__trial_01__train`.
- `checkpoint_last.pt` reached epoch `4` out of the requested `6`, with history tail showing validation loss continuing to move.
- The machine experienced unacceptable load, including a user-reported black-screen restart. This makes unbounded true-solver waiting an invalid operating mode on this workstation.

## Decision
- Keep the real cvxpy/cvxpylayers solver path. Do not weaken r50 into a surrogate just to make it cheap.
- Add resource governance around true-solver studies so the workstation remains usable and runs remain resumable.
- Default true-solver study profiles to safe resources. Full-machine execution must be an explicit choice, not the default.

## Implementation
- `run_self_optimizing_study.py` now supports `--resource-profile {auto,safe,balanced,full}`, `--thread-limit`, `--process-priority`, and `--cpu-affinity-count`.
- `auto` resolves r47/r50 true-solver profiles to `safe`; safe mode caps BLAS/OpenMP/Torch thread env vars, uses below-normal priority, and applies CPU affinity on Windows when possible.
- Progress files now record `resource_limits` and subprocess limit application results.
- `model_seq_v3.py` records thread-limit runtime details in training diagnostics via `resource_runtime`.
- Contract tests verify that r50 default auto is safe, full is explicit, loss profiles resolve, and progress events preserve resource metadata.

## Validation
- `py_compile` passed for `model_seq_v3.py`, `run_self_optimizing_study.py`, and `test_portfolio_daily_strategy_contracts.py`.
- Full `daily_research.continuous_policy.tests.test_portfolio_daily_strategy_contracts` passed: `72` tests OK.
- r50 safe dry-run confirms `resource_profile = safe`, `thread_limit = 4`, `process_priority = below_normal`, `cpu_affinity_mask = 15` on a 16 logical CPU machine.
- r50 full dry-run confirms `resource_profile = full`, `thread_limit = 0`, `cpu_affinity_mask = 0`, and `process_priority = normal`.

## Operating Rule
- After this event, do not run r50 true solver on this workstation without a resource profile.
- Use default/auto safe mode for normal research continuation.
- Use `--resource-profile full` only when the user explicitly accepts full-machine load risk.
- Prefer strict resume from the epoch-4 checkpoint rather than starting the same r50 smoke from scratch.

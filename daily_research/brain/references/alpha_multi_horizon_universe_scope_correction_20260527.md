# Alpha Multi-Horizon Universe Scope Correction - 2026-05-27

## Verdict

- Status: `scope_correction / evidence_boundary_repaired / shadow-only`.
- Affected studies: Stage 2.5, Stage 2.6, and Stage 2.7 multi-horizon stability runs.
- Decision: treat Stage 2.5-2.7 artifacts as `cap80_diagnostic`, not full rolling_liquid500 evidence.
- Stage 3 architecture review remains locked.
- No allocator, replay, paper, live/default, active artifact, or promotion action is authorized.

## What Happened

The recent Stage 2.5-2.7 drivers named the intended pool as `rolling_liquid500`, but the generated training commands did not bind the data-lake pool view and did not set `--max-universe-size 0`.

Because `run_alpha_path20_protocol.py` defaults `--max-universe-size` to `80`, those studies trained on a capped universe. Example evidence:

| Run | Universe scope | Universe size | Feature store shape | Train rows | Pool view |
|---|---|---:|---|---:|---|
| `path20_horizon_discovery_no_alpha_gru_liquid500_h1_2_3_5_8_10_15_20_30_du_cost20_hit10_dd010_20260523_01` | `full_rolling_liquid500` | `2430` | `[1699,2430,156]` | `452034` | `policy_pool_view__c11400fa72ad263f3d1eecfa` |
| `mh25_path_aux_fullgrid_rebudget_seed7_20260526_01` | `cap80_diagnostic` | `80` | `[1699,80,156]` | `74640` | empty |
| `mh26_score_monthly_robust_v1_fullgrid_seed7_20260527_01` | `cap80_diagnostic` | `80` | `[1699,80,156]` | `74640` | empty |
| `mh27_target_norm_head_constraint_v1_fullgrid_seed7_20260527_01` | `cap80_diagnostic` | `80` | `[1699,80,156]` | `74640` | empty |

The apparent training speedup is therefore explained mostly by sample count: the full-pool run had about `6.06x` more train rows than the recent cap80 runs, and observed epoch time moved from about `511s` to about `98s`.

## Interpretation

- The cap80 results are still useful as loss/target plumbing and relative diagnostic clues.
- They are not reliable evidence for full rolling_liquid500 stability, monthly negative-month behavior, or architecture readiness.
- Using cap80 to save compute can waste more compute later if it is mistaken for full-pool evidence, because the promising candidates must be rerun.

## Protocol Repair

Implemented code-side safeguards:

- Added `daily_research/path_policy/stage_universe_scope.py` to classify study artifacts as `full_rolling_liquid500`, `cap80_diagnostic`, or `partial_or_unknown`.
- Updated Stage 2.6 and Stage 2.7 drivers to bind full rolling_liquid500 commands with:
  - `--pool-view-kind rolling_liquidity`
  - `--pool-view-name rolling_liquid500`
  - `--max-universe-size 0`
  - `--forecast-memmap-manifest daily_research/output/path_policy/studies/path20_horizon_discovery_no_alpha_gru_liquid500_h1_2_3_5_8_10_15_20_30_du_cost20_hit10_dd010_20260523_01/forecast_dataset_manifest.json`
- Existing cap80 artifacts are no longer accepted as completed full-pool skips by the Stage 2.6 / Stage 2.7 training drivers.
- `load_forecast_memmap_dataset()` now resolves copied memmap manifests whose stored absolute paths point to an old workspace but matching files exist beside the manifest.

## Next Action

To answer the Stage 2.7 target/loss question at evidence grade, rerun on full rolling_liquid500 by reusing the existing full-pool memmap manifest rather than rebuilding the feature store.

Recommended rerun order:

1. Regenerate the Stage 2.7 task list and inspect that each command records `universe_scope=full_rolling_liquid500`.
2. Run a supervised full-pool rerun for `horizon_target_normalized_v1` and `target_norm_head_constraint_v1` first, seeds `7,11,19`.
3. Only add `horizon_head_soft_constraint_v1` if the first two candidates preserve positive rank/spread/hit but still need concentration diagnosis.
4. Keep the original gate unchanged: three seeds positive on rank/spread/hit, mean monthly positive rate `>=0.75`, max negative months `<=2`, and no worsened horizon concentration.

Full-pool reruns are expected to be roughly five times slower per epoch than the cap80 diagnostics. That is the correct cost for evidence-grade conclusions.


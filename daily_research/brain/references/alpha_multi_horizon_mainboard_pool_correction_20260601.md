# Alpha Multi-Horizon Mainboard Pool Correction - 2026-06-01

## Verdict

- Status: `mainboard_pool_correction_applied / research-only / shadow-only`.
- Mainline: `alpha_multi_horizon_utility_policy_v1`.
- User-confirmed universe contract: original research universe excludes ChiNext and STAR Market.
- Active artifact impact: none. `daily_research/output/active_execution_strategy.json` remains untouched.

## Problem

- The 2026-05-31 new-lineage rebuild used `policy_pool_view__8f9367ce305548e3ecd04feb`.
- That pool was `.SH/.SZ` only, but it did not exclude ChiNext or STAR Market prefixes.
- Therefore the rebuild anchor was useful diagnostics evidence, but it was not an old-research-equivalent universe.

## Evidence

Original new-lineage pool:

- Dataset id: `policy_pool_view__8f9367ce305548e3ecd04feb`.
- View: `rolling_liquid500`.
- Membership symbols: `5118`.
- Active universe size: `3668`.
- Daily member count min / median / max: `500 / 500 / 500`.
- Active excluded-board prefix counts: `300=754`, `301=252`, `688=332`, `689=0`; total `1338`.

Corrected mainboard pool:

- Dataset id: `policy_pool_view__74f45f4f83263bccd64a8027`.
- View: `rolling_liquid500_mainboard`.
- Source bundle: `policy_input_bundle__45e3d8c059ba718426a9f887`.
- Excluded prefixes: `300,301,688,689`.
- Membership symbols: `3175`.
- Active universe size: `2596`.
- Daily member count min / median / max: `500 / 500 / 500`.
- Active excluded-board prefix count: `0`.
- Prefix distribution after correction: `000=374`, `001=66`, `002=800`, `003=28`, `600=633`, `601=205`, `603=429`, `605=61`.

## Code Change

- `PoolViewSpec` now supports `exclude_symbol_prefixes`.
- `daily_research.data_lake.build_pool_view` exposes `--exclude-symbol-prefixes`.
- `daily_research.path_policy.run_alpha_path20_protocol` exposes `--pool-view-exclude-symbol-prefixes`.
- Unit coverage added for rolling liquidity pools that exclude `300/301/688/689` before ranking.

## Interpretation

- The prior rebuild result should be downgraded to `wrong_universe_diagnostic` for old-lineage comparison.
- The result difference is now more explainable: the rebuild included ChiNext and STAR Market while the original research contract excludes them.
- This does not prove the new data lake is bad. It shows the new rebuild did not match the intended research universe.

## Next Action

- Rebuild the alpha multi-horizon baseline using `policy_pool_view__74f45f4f83263bccd64a8027`.
- Keep execution frozen until a corrected mainboard-only, same-protocol candidate passes evidence-grade gates.
- Do not use `policy_pool_view__8f9367ce305548e3ecd04feb` as a short_v5b or Stage 2.8 equivalent comparison pool.

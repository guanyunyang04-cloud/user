# alpha_path20 Architecture Gain Result 2026-05-21

## Verdict
- Status: `forecast architecture-gain study / research / shadow-only / no promotion`.
- Mainline: `alpha_path20_neural_policy_v1`.
- Scope: controlled Path20 forecast architecture comparison on repaired data lake inputs.
- `promotion_allowed=false`, `shadow_only=true`, and `active_execution_strategy_expected_diff=none`.

## Inputs
- Lake dataset id: `policy_input_bundle__7c8f58d851bce8179e1e9e2d`.
- Sector/board view id: `policy_sector_board_view__4b07bb20fe1fbe05b6e13f45`.
- Feature profile: `raw_kline_context_sector_v1`.
- Static fields: `symbol,exchange,industry,board,liquidity_bucket,price_bucket`.
- Loss profile: `multitask_v1`.
- Selection profile: `multiscale`.
- Seed: `7`.
- Device: `cuda` with AMP.

## Completed Studies
- `path20_arch_gain_liquid500_sector_v1_20260520_01`: completed `forecast_walkforward_study`, verdict `forecast_test_confirmed`, pool view `policy_pool_view__c11400fa72ad263f3d1eecfa`, sample count `675876`, role counts train `456984`, validation `108920`, test `109972`.
- `path20_arch_gain_liquid800_sector_v1_20260520_01`: completed `forecast_walkforward_study`, verdict `forecast_test_confirmed`, pool view `policy_pool_view__8161f6879815d3a22dd3b3f2`, sample count `1079329`, role counts train `730053`, validation `173982`, test `175294`.

## Baselines
- `path20_pool_liquid500_sector_gru_seed7_probe_20260519_01`: GRU sector baseline, validation `rank_ic_20d=0.169761`, validation `top_bottom_spread_20d=0.045246`, test `rank_ic_20d=0.065573`, test `top_bottom_spread_20d=0.012952`.
- `path20_pool_liquid800_gru_seed7_probe_20260519_01`: GRU liquid800 baseline without sector static context, validation `rank_ic_20d=0.130004`, validation `top_bottom_spread_20d=0.031165`, test `rank_ic_20d=0.105120`, test `top_bottom_spread_20d=0.025195`.

## Liquid500 Results
| Family | Selected | Validation rank_ic_20d | Validation spread_20d | Test rank_ic_20d | Test spread_20d | Note |
|---|---:|---:|---:|---:|---:|---|
| `gru_sequence_static_context` | yes | 0.179232 | 0.041209 | 0.101635 | 0.019350 | Best selected family; improves test direction and spread versus sector GRU baseline. |
| `patch_transformer_static_context` | no | 0.159319 | 0.037778 | 0.048686 | 0.005670 | Positive validation/test direction, weaker than GRU static context. |
| `sector_slot_mixer_sequence` | no | 0.155676 | 0.033958 | 0.097630 | 0.017270 | Positive and close on test, but below GRU static context selection. |
| `stock_mixer_sequence` | no | 0.129180 | 0.029099 | 0.049329 | 0.012161 | Positive but weaker validation and test profile. |

## Liquid800 Results
| Family | Selected | Validation rank_ic_20d | Validation spread_20d | Test rank_ic_20d | Test spread_20d | Note |
|---|---:|---:|---:|---:|---:|---|
| `gru_sequence_static_context` | yes | 0.155924 | 0.037738 | 0.076870 | 0.012435 | Selected by validation; positive test, but below the old liquid800 GRU test baseline. |
| `sector_slot_mixer_sequence` | no | 0.158521 | 0.040360 | 0.033685 | 0.007147 | Highest validation rank among new families, but test transfer weak. |
| `patch_transformer_static_context` | no | 0.154207 | 0.037625 | 0.046018 | 0.007374 | Positive but below GRU static context on selected test metrics. |
| `stock_mixer_sequence` | no | 0.130310 | 0.025612 | 0.088608 | 0.022239 | Best liquid800 new-family test spread among non-selected models, but validation was weaker. |

## Diagnostics
- Slot diagnostics completed for both architecture-gain studies.
- `industry_available=true` and `board_available=true` for liquid500 and liquid800.
- `sector_slot_mixer_sequence` uses `dynamic_theme_factor_not_static_board` slot semantics; its validation gains did not consistently transfer to test.

## Interpretation
- Liquid500 confirms a useful input/architecture upgrade: static context on the GRU sequence model improves test `rank_ic_20d` and test spread versus the sector GRU baseline.
- Liquid800 confirms the new sector/static setup remains positive, but it does not beat the old liquid800 GRU probe on test `rank_ic_20d` or test spread.
- Patch transformer, stock mixer, and sector-slot mixer produced positive evidence, but this run does not prove they should replace GRU as the Path20 forecast anchor.
- The strongest current statement is: `gru_sequence_static_context` is the best controlled candidate on liquid500 sector data; broader liquid800 robustness is positive but not yet superior to the existing liquid800 GRU baseline.

## Boundaries
- This is forecast evidence only; no allocator, replay, oracle, live/default, or active execution claim.
- No production promotion is authorized.
- Do not change `daily_research/output/active_execution_strategy.json` based on this result.
- Do not treat dataset-only artifacts as completed forecast evidence.

## Next Allowed Actions
- Review why liquid800 static/sector context improved validation but weakened test versus the old liquid800 GRU baseline.
- Compare raw-context versus sector-context liquid800 under the same family and seed before adding more model families.
- Inspect selection-rule robustness: validation-selected `sector_slot_mixer_sequence` had weak liquid800 test transfer, while `stock_mixer_sequence` had stronger test spread but weaker validation.
- If training continues, prefer a narrow ablation on input/static context and selection rules before increasing seeds, universe size, or allocator complexity.

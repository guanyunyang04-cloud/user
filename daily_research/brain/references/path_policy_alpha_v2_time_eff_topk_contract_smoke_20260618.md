# Path Policy Alpha V2 Time-Efficient TopK Contract Smoke 20260618

## Verdict
- Status: `smoke_completed / code_path_validated / research_only / execution_frozen`.
- Run tag: `qdp_alpha_v2_time_eff_topk_contract_smoke_20260618_01`.
- Date: `2026-06-18`.
- Study root: `daily_research/output/path_policy/studies/qdp_alpha_v2_time_eff_topk_contract_smoke_20260618_01`.
- Active artifact impact: none. `daily_research/output/active_execution_strategy.json` remained absent and was not recreated.
- Interpretation: this smoke confirms the real QDP alpha_v2 label_v2 training pack can be consumed by the new `personal_time_efficient_topk_v1` contract. It is not model-quality evidence.

## Configuration
```text
training pack:
  quant_data_platform/data/memmap/training_pack/mainboard_style_structural_alpha_v2_label_v2_pack_full_2012_2025_20260616_01/qdp_training_pack_manifest.json

feature_profile:
  style_structural_alpha_v2

label_schema:
  path20_basic_v2 version 2

model:
  gru_sequence_static_context

loss_profile:
  personal_time_efficient_topk_v1

selection_profile:
  validation_loss

role samples:
  train 64 / validation 64 / test 64

device:
  cpu
```

## Result
- Protocol status: `completed`.
- Training summary status: `completed`.
- Selection rule: `lowest_validation_loss_for_training_loss_profile_then_seed_score`.
- Validation selection score: `-3.9466586112976074`.
- Validation `time_eff_profile_status`: `completed`.
- Test `time_eff_profile_status`: `completed`.
- Prediction CSVs include time-efficient columns such as:
  - `pred_time_eff_utility_1d`
  - `future_time_eff_utility_20d`
  - `pred_time_eff_score`
  - `future_time_eff_score`
  - `pred_time_eff_best_horizon`
  - `future_time_eff_best_horizon`

Metrics are not economically meaningful because this is a tiny 64/64/64 smoke. Rank IC and spread were mostly zero due to the intentionally small sampled role split.

## Command
```text
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.run_alpha_path20_protocol --stage forecast-walkforward-study --tag qdp_alpha_v2_time_eff_topk_contract_smoke_20260618_01 --data-source lake --lake-dataset-id policy_input_bundle__f926a496f69c61f6b92b5faf --execution-mode next_open --forecast-dataset-mode memmap --forecast-memmap-manifest quant_data_platform/data/memmap/training_pack/mainboard_style_structural_alpha_v2_label_v2_pack_full_2012_2025_20260616_01/qdp_training_pack_manifest.json --forecast-train-start-year 2012 --forecast-train-end-year 2023 --forecast-validation-year 2024 --forecast-test-year 2025 --forecast-model-families gru_sequence_static_context --forecast-max-feature-columns 384 --forecast-include-static-context --forecast-loss-profile personal_time_efficient_topk_v1 --forecast-selection-profile validation_loss --forecast-decision-cost-bps 20 --forecast-decision-hit-threshold-bps 10 --forecast-decision-drawdown-penalty 0.10 --forecast-cumulative-horizons 1,3,5,10,20 --forecast-horizon 20 --forecast-epochs 1 --forecast-min-epochs 1 --forecast-early-stop-patience 1 --forecast-batch-size 32 --forecast-hidden-dim 32 --forecast-gru-layers 1 --forecast-transformer-layers 1 --forecast-transformer-heads 1 --forecast-max-samples-per-role 64 --forecast-max-samples-per-date-per-role 16 --forecast-device cpu --no-forecast-amp --no-forecast-save-last --forecast-dataloader-num-workers 0
```

## Next Actions
1. Run Stage 0 old-model validation-loss checkpoint audit.
2. If audit does not reveal a higher-priority old-checkpoint issue, run the first real new-contract h256 seed7 full experiment:

```text
model_family=hybrid_expert_fusion_static_context
hidden_dim=256
transformer_layers=4
transformer_heads=8
gru_layers=2
batch_size=512
loss_profile=personal_time_efficient_topk_v1
selection_profile=validation_loss
```

3. Keep broad diagnostics separate from checkpoint selection:
   - `personal_topk_v1`
   - horizon rank/spread
   - same-candidate test
   - time-efficient score diagnostics

# Path Policy Alpha V2 Hybrid Alpha Score V2 Contract Smoke 20260621

## Verdict
- Status: `code_contract_implemented / focused_tests_passed / real_pack_smoked / research_only / execution_frozen`.
- Date: `2026-06-21`.
- Scope: add and validate `hybrid_alpha_score_v2` for `hybrid_structured_alpha_v2`.
- Active artifact impact: none. This did not touch `daily_research/output/active_execution_strategy.json`, live/default, paper/live/broker, score-backtest bridge, candidate matrix, or execution-candidate state.

## Contract

`hybrid_alpha_score_v2` is a prediction-first alpha-score loss. It is not an execution loss.

Allowed pairing:

```text
model_family: hybrid_structured_alpha_v2
output_profile: forecast_path_v1
loss_profile: hybrid_alpha_score_v2
static_context_fields: exchange,industry
```

Explicit boundaries:

```text
execution_cost_in_loss: false
execution_risk_penalty_in_loss: false
batch_topk_alignment_in_loss: false
decision_utility: 0.0
time_eff_topk_alignment: 0.0
promotion_allowed: false
```

Loss weights:

```text
path_daily:                0.58
quantile:                  0.18
path_aux:                  0.30
risk_aux:                  0.06
rank_aux:                  1.55
risk_rank_aux:             0.006
direction_aux:             0.010
downside_rank_aux:         0.012
alpha_efficiency_rank_aux: 0.85
decision_utility:          0.0
hit_aux:                   0.0
horizon_classification:    0.0
decision_rank_aux:         0.0
```

Profile-specific rank horizon weights:

```text
1d:  0.0100
3d:  0.0100
5d:  0.0100
10d: 0.0075
20d: 0.0050
```

Interpretation:
- v2 keeps `forecast_path_v1` and market-fact prediction as the direct training target.
- v2 increases rank and unit-time alpha efficiency pressure relative to `hybrid_alpha_score_v1`.
- v2 reduces the previous implicit long-horizon tilt by making 1/3/5d rank weights no weaker than 20d.
- Cost, risk utility, horizon choice, topK, and trade constraints remain external scorer/execution-layer concerns.

## Code Changes

Files changed:

```text
daily_research/path_policy/forecast_training.py
daily_research/path_policy/run_alpha_path20_protocol.py
daily_research/path_policy/tests/test_forecast_training.py
daily_research/path_policy/tests/test_rl_protocol.py
```

Key implementation points:
- Added `hybrid_alpha_score_v2` to forecast loss profiles.
- Added profile-aware rank loss weights via `_rank_loss_weights_for_profile(...)`.
- Kept `hybrid_alpha_score_v1` behavior unchanged.
- Added contract metadata for rank horizon bias and direct train target.
- Updated `hybrid_structured_alpha_v2` recommended losses to prefer `hybrid_alpha_score_v2`.
- Protocol accepts `hybrid_alpha_score_v2` for `hybrid_structured_alpha_v2` and still rejects execution-style losses.

## Real Pack Smoke

Run tag:

```text
qdp_alpha_v2_structured_alpha_v2_alpha_score_v2_contract_smoke_20260621_01
```

Training pack:

```text
quant_data_platform/data/memmap/training_pack/mainboard_style_structural_alpha_v2_label_v2_pack_full_2012_2025_20260616_01/qdp_training_pack_manifest.json
```

Smoke config:

```text
model_family: hybrid_structured_alpha_v2
loss_profile: hybrid_alpha_score_v2
output_profile: forecast_path_v1
static fields: exchange,industry
max_samples_per_role: 32
device: cpu
hidden_dim: 24
batch_size: 8
epochs: 1
```

Result:

```text
status: completed
selected_model_family: hybrid_structured_alpha_v2
selected_seed: 7
best_epoch: 1
best_validation_loss: 3.023409605026245
train_rows: 32
validation_rows: 32
test_rows: 32
```

This smoke only proves code/data compatibility and the training contract path. It is not model-quality evidence.

## Verification

Focused tests:

```text
11 passed in 12.69s
```

Covered:
- `hybrid_alpha_score_v2` contract is prediction-first and unit-time-aware.
- `hybrid_structured_alpha_v2` trains with `hybrid_alpha_score_v2`.
- Protocol accepts v2 and still rejects symbol static context and execution loss.
- Existing structured alpha v2 model contract still passes.

Additional checks completed before writeback:

```text
py_compile forecast_training.py run_alpha_path20_protocol.py: passed
active_execution_strategy diff: empty
```

## Next Allowed Action

The next training action, when requested, should use:

```text
hybrid_structured_alpha_v2
hybrid_alpha_score_v2
forecast_path_v1
exchange,industry
hidden_dim=192
batch_size=128
```

No `h192` no-stride controlled scout was launched in this step, per user request.

The runtime choice remains open:
- `train_date_stride=1` preserves all train dates but is expensive.
- `train_date_stride=3` remains a diagnostic shortcut, not the only valid training semantics.

No multi-seed, score-backtest bridge, candidate matrix, execution-candidate review, paper/live/broker, or active/default artifact change is authorized by this evidence.

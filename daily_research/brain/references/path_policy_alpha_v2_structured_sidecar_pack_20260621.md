# Path Policy Alpha V2 Structured Sidecar Pack 20260621

## Verdict
- Status: `sidecar_pack_completed / model_contract_validated / research_only / execution_frozen`.
- Date: `2026-06-21`.
- Scope: build a `structured_alpha_v2` dedicated sidecar training pack for `hybrid_structured_alpha_v2`.
- Active artifact impact: none. This did not touch `daily_research/output/active_execution_strategy.json`, live/default, paper/live/broker, QDP active registry pointers, score-backtest bridge, candidate matrix, or execution-candidate state.

## Purpose

The previous full QDP training pack is semantically usable, but not model-aligned for `hybrid_structured_alpha_v2`:

```text
source static context: symbol,exchange,industry
source feature layout: stock_date_feature with original alpha_v2 column order
model contract: no symbol_id; use exchange,industry; consume structured feature groups
```

The new sidecar keeps the original alpha_v2 label_v2 sample/label contract, but makes the model-facing contract explicit:

```text
feature columns: structured_alpha_v2 group-contiguous order
static context: exchange,industry only
labels: reused source sample-major label arrays by absolute path
window materialization: runtime sliding window view
full rolling-window cache: false
```

This avoids materializing all `252d x 307` rolling windows, which would be roughly terabyte-scale.

## Artifact

Source pack:

```text
quant_data_platform/data/memmap/training_pack/mainboard_style_structural_alpha_v2_label_v2_pack_full_2012_2025_20260616_01/qdp_training_pack_manifest.json
```

Sidecar manifest:

```text
quant_data_platform/data/memmap/training_pack/mainboard_style_structural_alpha_v2_label_v2_structured_alpha_v2_pack_20260621_01/qdp_structured_alpha_v2_training_pack_manifest.json
```

Sidecar feature panel:

```text
quant_data_platform/data/memmap/training_pack/mainboard_style_structural_alpha_v2_label_v2_structured_alpha_v2_pack_20260621_01/feature_panel_stock_date_feature_structured_alpha_v2.float16.dat
```

Size/cost:

```text
feature panel: 7,715,124,900 bytes
sample_count: 7,421,714
train: 6,050,268
validation: 681,979
test: 689,467
feature_count: 307
```

Feature group counts used by the sidecar manifest:

```text
daily_price_volume:    65
cross_section:         19
market_regime:         25
industry_peer:         24
valuation_liquidity:   33
event_quality:         15
intraday:             126
```

Static context:

```text
fields: exchange,industry
static_context_ids shape: [7421714, 2]
symbol_id: removed
```

## Code Contract

Implemented in:

```text
daily_research/path_policy/forecast_dataset.py
daily_research/path_policy/forecast_training.py
daily_research/path_policy/tests/test_forecast_memmap_dataset.py
daily_research/path_policy/tests/test_forecast_training.py
```

Key changes:
- Added `build_qdp_structured_alpha_v2_training_pack(...)`.
- Added sidecar-aware path resolution for copied/derived manifests.
- Sidecar builder rejects `date_major=True` for now; the supported training artifact is stock-major only.
- Sidecar builder requires source static IDs and writes `static_context_ids_exchange_industry.int32.dat`.
- Training now prioritizes `manifest["structured_alpha_v2_pack"]["feature_group_indices"]` for `hybrid_structured_alpha_v2`.
- If sidecar metadata is absent, training falls back to column-name feature grouping for backward compatibility.
- Checkpoints and summaries record `feature_group_source`.

Important fix during implementation:
- Initial smoke proved the model could run, but also showed training was still recomputing groups from column names.
- This was fixed so `hybrid_structured_alpha_v2` now uses `feature_group_source=manifest_structured_alpha_v2_pack` when the sidecar manifest is used.

## Real Smoke

Run tag:

```text
qdp_alpha_v2_structured_sidecar_hybrid_alpha_score_v2_smoke_20260621_02
```

Study root:

```text
daily_research/output/path_policy/studies/qdp_alpha_v2_structured_sidecar_hybrid_alpha_score_v2_smoke_20260621_02
```

Smoke config:

```text
model_family: hybrid_structured_alpha_v2
loss_profile: hybrid_alpha_score_v2
output_profile: forecast_path_v1
selection_profile: validation_loss
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
best_validation_loss: 3.0253596901893616
train_loss: 2.2549378275871277
train_samples_per_second: 32.52032520335976
dataset_type: ForecastTrainingPackDataset
```

Checkpoint contract:

```text
feature_group_source: manifest_structured_alpha_v2_pack
static_context_fields: exchange,industry
feature_group_counts:
  daily_price_volume: 65
  cross_section: 19
  market_regime: 25
  industry_peer: 24
  valuation_liquidity: 33
  event_quality: 15
  intraday: 126
```

This is code/data/model contract evidence only. It is not model-quality evidence.

## Verification

Focused tests:

```text
2 passed in 13.02s
```

Covered:
- Sidecar feature reorder, label reuse, and `exchange/industry` static context.
- Training uses sidecar manifest feature groups instead of recomputing groups from column names.

Additional real checks:

```text
py_compile forecast_dataset.py forecast_training.py test_forecast_memmap_dataset.py test_forecast_training.py: passed
real sidecar load smoke: passed
real sidecar training smoke: completed
```

## Next Allowed Action

Next controlled experiment should explicitly use the sidecar manifest:

```text
--forecast-memmap-manifest quant_data_platform/data/memmap/training_pack/mainboard_style_structural_alpha_v2_label_v2_structured_alpha_v2_pack_20260621_01/qdp_structured_alpha_v2_training_pack_manifest.json
--forecast-model-families hybrid_structured_alpha_v2
--forecast-loss-profile hybrid_alpha_score_v2
--forecast-output-profile forecast_path_v1
--forecast-train-static-fields exchange,industry
--forecast-hidden-dim 192
--forecast-batch-size 128
--forecast-dataloader-num-workers 0
--no-forecast-per-epoch-prediction-metrics
```

Runtime choice remains explicit:
- `train_date_stride=1` keeps all train dates and is semantically clean but expensive.
- `train_date_stride=3` is still a controlled diagnostic/runtime shortcut with validation/test full.

Do not use the older source pack directly for `hybrid_structured_alpha_v2` formal experiments unless the intent is a backward-compatibility comparison. The sidecar is now the model-aligned pack.

## Boundaries

- No full h192 training launched here.
- No multi-seed, score-backtest bridge, candidate matrix, execution-candidate review, paper/live/broker, or active/default artifact change is authorized by this evidence.
- QDP registry pointers remain unchanged; this sidecar is explicit-manifest research data.

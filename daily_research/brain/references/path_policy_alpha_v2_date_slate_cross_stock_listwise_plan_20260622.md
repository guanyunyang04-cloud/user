# Path Policy Alpha V2 Date-Slate Cross-Stock Listwise Plan 20260622

## Verdict
- Status: `implementation_in_progress / code_contract_partially_validated / research_only / execution_frozen`.
- Date: `2026-06-22`.
- Research program: `qdp_alpha_v2_generalization_repair`.
- Scope: supersede the immediate next date-slate architecture step from `date_slate_alpha_fusion_v1` to a model-level cross-stock/listwise slate line.
- Active artifact impact: none. This work does not touch `daily_research/output/active_execution_strategy.json`, live/default, paper/live/broker, score-backtest bridge, candidate matrix, execution-candidate state, QDP canonical data, or QDP registry active pointers.

## Why This Supersedes The Prior Date-Slate Next Step

`date_slate_alpha_fusion_v1` fixed the biggest old mismatch: loss ranking is now same-date grouped instead of arbitrary stock-major batch ranking. But the model forward itself still predicts each stock independently:

```text
one stock 252d history -> per-stock fused token -> incremental path head
same-date comparison only appears in the loss
```

For personal small-capital top-K selection, that is not the full intended semantic. The desired model should directly see same-date stock context before producing alpha scores. This is full-market only when `stocks_per_date` covers the whole trainable date universe; otherwise it is explicitly sampled-chunk context:

```text
one stock 252d history
+ same-date chunk or full-date slate context
-> future incremental alpha path / rankable alpha score
```

The new line therefore adds a model-level same-date cross-stock slate mixer while staying prediction-first. It does not yet implement explicit market slots, industry slots, similar-stock slots, or full self-attention.

## New Model Family

Implemented family:

```text
date_slate_cross_stock_alpha_fusion_v1
```

Contract:

```text
output_profile: forecast_incremental_path_v2
loss_profile: date_listwise_alpha_score_v2
static context: exchange,industry only
symbol_id: forbidden
dataset: QDP date-slate training pack with date-major feature panel
scope: research-only
```

Architecture:

```text
per-stock date_slate_alpha_fusion_v1 temporal/group/intraday backbone
  -> base fused token [N, 4H]
  -> same-date low-rank learned-slot mixer using date_group_ids
  -> cross residual
  -> residual gate
  -> enhanced fused token = base_fused + gate * residual
  -> forecast_incremental_path_v2 head
```

Important semantics:

```text
1. Cross-stock context is grouped strictly by date_group_ids.
2. No stock from date A can attend to date B.
3. Cross-stock mixer is residual and gated, so the single-stock backbone remains protected.
4. The mixer is low-rank/slot based, not naive N^2 full attention. This is mainly justified for full-date scalability; on small sampled chunks it should be treated as a latent-factor bottleneck/regularizer, not a compute necessity.
5. If training uses stocks_per_date below the full same-date stock count, the result is sampled-slate context, not full-market context; this is recorded in training_config.date_slate_semantics and must not be interpreted as full-market listwise evidence.
6. For sampled-chunk experiments, a direct full self-attention chunk mixer is a more semantically honest ablation than presenting low-rank slots as full-market context.
```

## New Loss Profile

Implemented profile:

```text
date_listwise_alpha_score_v2
```

This is not an execution utility loss. It is still prediction-first and market-fact oriented. Despite the historical name, it is a same-date grouped/listwise proxy, not strict full-slate ListMLE/ListNet/NDCG and not full-market top-K unless the date slate itself is full-date.

Weights:

```text
path_daily:           0.45
quantile:             0.12
path_aux:             0.20
risk_aux:             0.04
date_grouped_rank:    1.00
unit_time_alpha_rank: 0.90
listwise_soft_topk:   0.65
downside_rank_aux:    0.008
upside_rank_aux:      0.006
base_path_aux:        0.18
base_rank_aux:        0.25
cross_residual_reg:   0.010
cross_gate_reg:       0.006
decision_utility:     0.0
```

Meaning:

```text
date_grouped_rank: same-date cumulative alpha ordering
unit_time_alpha_rank: same-date cumulative alpha / horizon ordering
listwise_soft_topk: differentiable same-date topK-oriented alpha gain
base_path_aux/base_rank_aux: keep the per-stock backbone predictive
cross_residual_reg/cross_gate_reg: prevent slate context from blindly overwriting base prediction
```

This directly addresses the user requirement that model training and checkpoint selection follow the same objective, while avoiding premature execution-cost/portfolio-state learning.

## Implemented Code Contract So Far

Changed files:

```text
daily_research/path_policy/models.py
daily_research/path_policy/forecast_training.py
daily_research/path_policy/run_alpha_path20_protocol.py
daily_research/path_policy/tests/test_models.py
daily_research/path_policy/tests/test_forecast_training.py
daily_research/path_policy/tests/test_rl_protocol.py
```

Key implementation points:

```text
DateSlateAlphaFusionV1Forecaster now exposes reusable base fused state helpers.
DateSlateCrossStockAlphaFusionV1Forecaster adds same-date low-rank slot residual mixer.
_forecast_model_forward can pass date_group_ids to models that accept it.
date-slate train/eval loss paths pass date_group_ids to the model.
_predict_indices uses date-slate prediction for date_group_ids-aware models; if training was sampled-chunk and prediction uses a fuller date slate, that is a recorded scope mismatch rather than a proven-equivalent mode.
Protocol guard requires date_slate_cross_stock_alpha_fusion_v1 + date_listwise_alpha_score_v2.
Old date_slate_alpha_fusion_v1 remains constrained to date_grouped_alpha_score_v1.
```

Focused tests passed:

```text
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest \
  daily_research/path_policy/tests/test_models.py::test_date_slate_cross_stock_alpha_fusion_v1_uses_same_date_context_only \
  daily_research/path_policy/tests/test_forecast_training.py::test_make_forecast_model_registers_date_slate_alpha_fusion_v1 \
  daily_research/path_policy/tests/test_forecast_training.py::test_date_listwise_alpha_score_contract_and_loss_uses_cross_stock_outputs \
  daily_research/path_policy/tests/test_forecast_training.py::test_train_forecast_models_date_slate_alpha_fusion_v1_end_to_end \
  daily_research/path_policy/tests/test_rl_protocol.py::test_protocol_accepts_date_slate_cross_stock_alpha_fusion_v1_contract -q

5 passed, 48 warnings
```

Warnings are from tiny synthetic constant inputs in metric correlation and are not model-contract failures.

## Remaining Before Formal Training

Required next checks:

```text
1. Run broader focused path_policy tests covering models / forecast_training / protocol fast contracts.
2. Run py_compile on changed Python files.
3. Run git diff --check.
4. Run doc_guard changed scope and brain integrity check.
5. Run a real-pack CPU smoke for date_slate_cross_stock_alpha_fusion_v1 + date_listwise_alpha_score_v2.
6. Run CUDA throughput scout before any long training.
```

Recommended first real-pack smoke:

```text
model_family: date_slate_cross_stock_alpha_fusion_v1
output_profile: forecast_incremental_path_v2
loss_profile: date_listwise_alpha_score_v2
hidden_dim: 24 or 32
role caps: 16/16/16 or 32/32/32
dates_per_batch: 1
stocks_per_date: large enough for full smoke date if possible
static fields: exchange,industry
selection_profile: validation_loss
finite_guard: enabled
per_epoch_prediction_metrics: disabled
device: CPU
```

Recommended first CUDA throughput scout:

```text
hidden_dim: 128
dates_per_batch: 1
stocks_per_date: 256, then 512/1024/4096 only if memory permits
AMP: off initially
finite_guard: enabled
selection_profile: validation_loss
per_epoch_prediction_metrics: disabled
```

## Explicit Boundaries

This is not yet model-quality evidence.

Not authorized from this reference alone:

```text
multi-seed
score-backtest bridge
candidate matrix
execution-candidate review
paper/live/broker
active/default artifact changes
QDP registry pointer changes
deleting old data assets
```

## Current Next Action

Finish validation of the code contract, then run real-pack smoke. If smoke and finite checks pass, run CUDA throughput scout to decide whether full-date `stocks_per_date=4096` is feasible. If full-date is not feasible, split the path into `sampled_chunk_full_self_attention` and `sampled_chunk_low_rank_slot` diagnostics, and do not label either as full-market.

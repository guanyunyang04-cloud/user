# Path Policy Alpha V2 Date-Slate Incremental Architecture Plan 20260621

## Verdict
- Status: `superseding_plan / implementation_not_started / research_only / execution_frozen`.
- Date: `2026-06-21`.
- Research program: `qdp_alpha_v2_generalization_repair`.
- Scope: replace the current hot path from `hybrid_structured_alpha_v2 h192 stock-major no-stride full` to a new date-slate, incremental-output, prediction-first alpha model line.
- Active artifact impact: none. This plan does not touch `daily_research/output/active_execution_strategy.json`, live/default, paper/live/broker, score-backtest bridge, candidate matrix, execution-candidate state, or QDP registry active pointers.

## Why This Supersedes The Prior Hot Path

The previous structured sidecar and `hybrid_structured_alpha_v2` work established useful contracts:

```text
feature profile: style_structural_alpha_v2
label schema: path20_basic_v2
feature count: 307
static context: exchange,industry
symbol_id: disabled
feature groups: manifest-defined group-contiguous layout
```

However, the next formal model line should not continue as a simple h192 stock-major full run because the current system has unresolved structural issues:

```text
1. Rank loss is batch-internal, but stock-major batches are not same-date cross sections.
2. forecast_path_v1 exposes daily and cumulative outputs separately, so intraday residual on cumulative aux is not fully clean.
3. Router currently multiplies expert tokens before fusion, which can weaken low-weight experts and encourage expert death.
4. Intraday/adjust features can contain extreme normalized values; the failed h192 no-stride run produced all-NaN checkpoints.
5. The effective independent sample count is closer to 2,894 train market dates plus finite market regimes, not 6.05M IID rows.
```

The new plan keeps the high-level intent from the existing architecture:

```text
semantic feature grouping
context-conditioned prediction
multi-scale time experts
intraday as short-horizon residual
prediction-first alpha score generation
```

but changes the training contract and model output to make the semantics cleaner.

## Current Evidence Baseline

Relevant completed assets:

```text
structured sidecar manifest:
quant_data_platform/data/memmap/training_pack/mainboard_style_structural_alpha_v2_label_v2_structured_alpha_v2_pack_20260621_01/qdp_structured_alpha_v2_training_pack_manifest.json

structured sidecar reference:
daily_research/brain/references/path_policy_alpha_v2_structured_sidecar_pack_20260621.md
```

Current structured sidecar counts:

```text
train rows:       6,050,268
validation rows:    681,979
test rows:          689,467
train dates:          2,894
validation dates:       221
test dates:             222
avg train rows/date: ~2,091
lookback:              252
horizon:                20
feature_count:         307
```

Current feature groups:

```text
daily_price_volume:    65
cross_section:         19
market_regime:         25
industry_peer:         24
valuation_liquidity:   33
event_quality:         15
intraday:             126
```

Current model parameter scale from the implemented `hybrid_structured_alpha_v2`:

```text
D=96   heads=4  -> 1.91M params
D=128  heads=4  -> 3.30M params
D=192  heads=6  -> 7.19M params
D=256  heads=8  -> 12.58M params
```

The new default capacity should be `D=128`; `D=192` remains a large candidate after the new contract is stable.

## Design Principles

### 1. Prediction First, Not Execution Model

The model should predict market facts and stable alpha scores:

```text
future daily excess return increments
cumulative returns derived from increments
rankable alpha strength
unit-time alpha efficiency
future downside/upside/worst-day facts
quantile uncertainty
```

The model should not learn execution rules as its primary target:

```text
no direct transaction-cost utility target in main loss
no portfolio Sharpe/turnover/capacity objective in main loss
no live/default or execution-candidate behavior
```

Cost, turnover, liquidity constraints, holding rules, and final top-K execution preference remain in scorer/backtest/execution layers.

### 2. Date-Slate Rank Semantics

Rank loss must compare stocks on the same prediction date.

New training batches should carry date structure:

```text
batch = D_dates x N_stocks_per_date
forward input flattened as [B,252,307]
loss sees date_id/slate_id and computes rank terms only within each date group
```

Recommended first training shapes:

```text
1 date x 512 stocks
2 dates x 256 stocks
4 dates x 128 stocks
```

If a date has more stocks than the slate size, it can be split into multiple random or deterministic slate chunks. This is not perfect full-market listwise ranking, but it is semantically cleaner than cross-date batch rank and computationally feasible.

### 3. New Date-Slate Sidecar Pack

A dedicated pack is allowed and recommended.

Do not materialize all rolling windows:

```text
7.4M samples x 252 x 307 x float16 ~= 1.1TB
```

Instead create a date-major model-ready sidecar:

```text
date_major_feature_panel: date x stock x feature, float16
date_slate_index: date -> sample row ids / stock positions / role
static_context_ids: exchange,industry
labels: reuse or reference source sample-major label arrays
feature_group_indices: carry structured_alpha_v2 manifest groups
```

Expected added storage:

```text
date-major feature panel: about 7.7GB
index/manifests/diagnostics: small
total expected: about 8-10GB
```

This is acceptable compared with long training time waste from inefficient stock-major date-slate access.

### 4. Numeric Stability Monitoring Only At First

Do not change input values during the first implementation unless a later diagnostic requires it.

Initial numeric stability policy:

```text
no clipping
no winsorization
no canonical data changes
no model-ready value alteration
```

Add only fail-fast monitoring:

```text
input finite check
prediction finite check
loss finite check
gradient finite check
parameter finite check
AMP scaler state recording
bad batch dump
```

When a non-finite value is found, training must stop immediately and write enough evidence:

```text
run tag
epoch
step
role
row_ids
dates
stocks
feature absmax by group
top offending features
label stats
prediction stats if available
loss component values if available
AMP scale
```

This preserves training semantics while preventing another all-NaN checkpoint.

### 5. Incremental Output Contract

Create a new output profile instead of forcing the old `forecast_path_v1` semantics.

Proposed output profile:

```text
forecast_incremental_path_v2
```

Core outputs:

```text
daily_mu:        [B,20]       future daily excess-return increments
daily_q10/q50/q90: [B,20]    daily increment quantiles
cum_mu_by_horizon: [B,5]     derived from daily_mu cumulative sums at 1/3/5/10/20
risk_aux:        [B,5,3]     drawdown / worst / upside facts
alpha_score_aux: optional derived or explicit alpha score channels
diagnostics: router weights, group weights, context gates, intraday residual norm
```

For the first version, cumulative predictions should be derived from daily increments:

```text
cum_mu_h = sum(daily_mu_1 ... daily_mu_h)
```

This makes intraday residual semantics clean:

```text
daily_mu_k = base_daily_mu_k + gate_k * intraday_residual_k
```

The residual can decay across future daily increments, but cumulative outputs then naturally include the residual contribution through summation.

### 6. New Model Family

Proposed model family name:

```text
date_slate_alpha_fusion_v1
```

Default capacity:

```text
d_model: 128
heads: 4
ffn: 256
dropout: 0.10
head_dropout: 0.15
static fields: exchange,industry
symbol_id: forbidden
```

Architecture:

```text
input [B,252,307]
  -> split 7 feature groups
  -> six daily groups through independent group encoders
  -> within-day group attention, preserving group axis
  -> main sequence Q [B,252,D]
  -> group summaries [B,6,D]
  -> explicit context encoder
  -> residual zero-init FiLM
  -> 3 time experts:
       local/mid-term TCN
       recent patch Transformer
       learnable/fixed-initial EWMA
  -> expert-id + context-aware fusion
  -> incremental distributional forecast head
  -> context-gated intraday residual on daily increments
  -> derived cumulative outputs
```

Initial expert set should be three experts, not four:

```text
TCN: local/mid-term price path and volatility shape
Patch Transformer: recent multi-scale non-local structure
EWMA: stable information-decay trend
```

GRU is deferred. It can be reintroduced as a large/ablation candidate later, but the first v1 should avoid overlapping recurrence cost while the contract is being stabilized.

### 7. Context And Fusion Rules

Context encoder should use explicit group summaries:

```text
market_regime summary
industry_peer summary
event_quality summary
valuation_liquidity summary
recent Q summary
static exchange/industry summary
```

FiLM should be residual and weak at initialization:

```text
Z_t = Q_t + rho_t * (gamma(context) * LN(Q_t) + beta(context))
```

Recommended:

```text
gamma/beta last layer zero-init
gamma constrained with tanh or small scale
rho_t recency-biased but bounded
```

Router/fusion should not hard-suppress expert tokens before fusion.

Avoid:

```text
weighted = tokens * router_weights
fusion(weighted)
```

Prefer:

```text
tokens_with_id = expert_tokens + expert_id_embeddings
fusion_input = tokens_with_id plus context token
router weights used as residual gate or final pooling bias
gated_token = token * (1 + alpha * router_weight)
```

The goal is to avoid early expert death while still allowing context-aware specialization.

### 8. Intraday Branch Rules

Intraday remains a short-horizon residual branch, not a main predictor.

Initial semantics:

```text
intraday input: [B,252,126] from structured pack
intraday encoder: recency pooling or shallow TCN over recent window
context gate: market/liquidity/event context controls trust in intraday
residual output: daily increment residual [B,20]
horizon residual mask: strongest at day 1, decays over future increments
```

Do not clip intraday at first, per numeric policy. But monitor:

```text
intraday input absmax
intraday residual absmax
intraday residual norm
intraday contribution / base prediction ratio
```

If fail-fast finds intraday values as the first non-finite driver, a later plan may introduce model-ready clipping or local float32 protection.

## Loss Contract

Proposed loss profile:

```text
date_grouped_alpha_score_v1
```

The loss is prediction-first:

```text
daily path Huber
daily quantile pinball
derived cumulative Huber
date-grouped rank loss
unit-time alpha efficiency rank loss
risk auxiliary Huber
downside/upside rank small auxiliary
```

No execution utility in v1:

```text
decision_utility: 0
hit_aux: 0
transaction cost: not in main loss
portfolio turnover/capacity: not in main loss
```

Rank terms:

```text
For each date group in the batch:
  compute score_h = cum_mu_h or cum_mu_h / h
  compare only stocks within that date group
  skip groups smaller than min_rank_group_size
  cap pair count per date for speed
```

Suggested initial weights:

```text
daily_path:                 0.60
quantile:                   0.18
derived_cum_path:           0.25
risk_aux:                   0.06
date_grouped_rank:          1.20
unit_time_alpha_rank:       0.70
downside_rank:              0.010
upside_rank:                0.006
router_entropy_monitor:     0.0 initially
expert_diversity_monitor:   0.0 initially
```

Warmup policy:

```text
epoch 1 or first warmup steps: path + quantile + risk
after warmup: add date-grouped rank and unit-time rank
```

Dynamic loss balancing is not required in the first implementation, but the code should log component losses so EMA normalization or GradNorm can be added later with evidence.

## Date-Slate Pack Plan

New artifact family:

```text
qdp_date_slate_alpha_v2_pack_v1
```

Suggested root:

```text
quant_data_platform/data/memmap/training_pack/mainboard_style_structural_alpha_v2_label_v2_date_slate_pack_YYYYMMDD_01
```

Inputs:

```text
source structured sidecar manifest:
quant_data_platform/data/memmap/training_pack/mainboard_style_structural_alpha_v2_label_v2_structured_alpha_v2_pack_20260621_01/qdp_structured_alpha_v2_training_pack_manifest.json
```

Outputs:

```text
qdp_date_slate_training_pack_manifest.json
feature_panel_date_stock_feature.float16.dat
date_slate_index.parquet
date_values.json or embedded manifest list
stock_values.json or embedded manifest list
static_context_ids_exchange_industry.int32.dat, or reference if shape-compatible
label array references to source pack
```

Manifest fields:

```text
artifact_type: qdp_date_slate_training_pack_v1
source_manifest_json
source_artifact_type
feature_layout: date_stock_feature
feature_shape: [date_count, stock_count, feature_count]
feature_dtype: float16
lookback_days: 252
horizon: 20
cumulative_horizons: [1,3,5,10,20]
feature_group_indices
static_context_schema
sample_count_by_role
date_count_by_role
date_slate_contract
```

`date_slate_index.parquet` should include:

```text
sample_row_idx
role
date
global_date_pos
stock
global_stock_pos
has_full_lookback
label_end_date
```

The loader should resolve a batch by date:

```text
date_pos -> source_start = date_pos - lookback + 1
selected stock positions for that date
read panel[source_start:date_pos+1, selected_stocks, :]
transpose to [selected_stocks, lookback, feature]
return x, labels, static_ids, row_ids, date_ids
```

## Implementation Phases

### Phase 0: Brain/Contract Lock

Goal: ensure future agents follow this plan.

Tasks:

```text
write this reference
update state_center.md P3c-now
register reference in evidence_registry.json
run doc guard / brain integrity
```

### Phase 1: Numeric Fail-Fast Diagnostics

Goal: prevent silent all-NaN checkpoints without changing model semantics.

Files likely touched:

```text
daily_research/path_policy/forecast_training.py
daily_research/path_policy/tests/test_forecast_training.py
```

Implement:

```text
finite_guard_config
bad_batch_dump writer
loss component logging hook
prediction finite check
loss finite check
grad finite check
param finite check
```

Expected behavior:

```text
If non-finite value appears, raise a clear exception and write a diagnostic JSON.
No clipping, winsorization, or tensor value replacement.
```

### Phase 2: Date-Slate Sidecar Pack

Goal: build date-major pack and slate index.

Files likely touched:

```text
daily_research/path_policy/forecast_dataset.py
daily_research/path_policy/tests/test_forecast_memmap_dataset.py
```

Implement:

```text
build_qdp_date_slate_training_pack(...)
load date-slate manifest
validate date-major panel size/dtype
reuse source label arrays
copy/reuse exchange,industry static ids
write date_slate_index.parquet
```

Verification:

```text
small synthetic pack test
real source sidecar build smoke
manifest json.tool
load first train date slate
finite check on sample batch
```

### Phase 3: Date-Slate Dataset And Loader

Goal: produce training batches with date structure.

Files likely touched:

```text
daily_research/path_policy/forecast_dataset.py
daily_research/path_policy/forecast_training.py
daily_research/path_policy/tests/test_forecast_memmap_dataset.py
```

Implement:

```text
ForecastDateSlateTrainingPackDataset
date_slate_torch_dataset(...)
sampler options:
  dates_per_batch
  stocks_per_date
  shuffle_dates
  shuffle_stocks_within_date
  deterministic_seed
return:
  x [B,252,307]
  y_daily [B,20]
  y_cum [B,5]
  y_risk [B,5,3]
  row_ids [B]
  date_ids [B]
  date_group_offsets or date_group_index
  static_ids [B,2]
```

### Phase 4: New Incremental Output Profile

Goal: introduce `forecast_incremental_path_v2`.

Files likely touched:

```text
daily_research/path_policy/models.py
daily_research/path_policy/tests/test_models.py
```

Implement:

```text
path20_incremental_output_dim(...)
split_incremental_path20_outputs(...)
derived cumulative outputs from daily_mu
quantile outputs for daily increments
risk aux outputs by horizon
```

Compatibility:

```text
Do not break forecast_path_v1.
Existing checkpoints remain loadable under old families.
```

### Phase 5: New Model Family

Goal: implement `date_slate_alpha_fusion_v1`.

Files likely touched:

```text
daily_research/path_policy/models.py
daily_research/path_policy/forecast_training.py
daily_research/path_policy/tests/test_models.py
daily_research/path_policy/tests/test_forecast_training.py
```

Implement modules:

```text
group encoders with preserved group axis
within-day group attention
explicit context encoder
residual zero-init FiLM
TCN expert
recent patch expert
EWMA expert
expert-id + context fusion
incremental forecast head
context-gated intraday residual
diagnostics outputs
```

Model contract:

```text
symbol static context forbidden
static fields must be exchange,industry
default hidden_dim=128
output_profile must be forecast_incremental_path_v2
loss_profile should be date_grouped_alpha_score_v1
```

### Phase 6: Date-Grouped Loss

Goal: implement `date_grouped_alpha_score_v1`.

Files likely touched:

```text
daily_research/path_policy/forecast_training.py
daily_research/path_policy/tests/test_forecast_training.py
```

Implement:

```text
date_grouped_pairwise_rank_loss(score, target, date_ids, min_group_size, max_pairs_per_date)
unit_time_alpha_rank_loss using derived cum_mu / horizon
component loss dictionary
warmup schedule
date-group skip accounting
```

Must not compare stocks across different dates.

### Phase 7: Protocol Wiring

Goal: expose the new line through `run_alpha_path20_protocol.py`.

Files likely touched:

```text
daily_research/path_policy/run_alpha_path20_protocol.py
daily_research/path_policy/forecast_training.py
daily_research/path_policy/tests/test_rl_protocol.py
```

Add CLI options:

```text
--forecast-model-families date_slate_alpha_fusion_v1
--forecast-output-profile forecast_incremental_path_v2
--forecast-loss-profile date_grouped_alpha_score_v1
--forecast-date-slate-pack
--forecast-dates-per-batch
--forecast-stocks-per-date
--forecast-rank-min-group-size
--forecast-rank-max-pairs-per-date
--forecast-finite-guard
--forecast-bad-batch-dump-dir
```

### Phase 8: Smoke And Throughput

Goal: prove code/data/model/loss contracts before any long run.

Required checks:

```text
unit tests
py_compile changed files
synthetic date-slate pack load
real date-slate pack build smoke
real model smoke, CPU, tiny role caps
real CUDA throughput scout
finite guard negative test
checkpoint save/load restore
```

Suggested CUDA scout configs:

```text
D=128
dates_per_batch=1, stocks_per_date=256
dates_per_batch=1, stocks_per_date=512
dates_per_batch=2, stocks_per_date=256
workers=0 first
AMP on/off scout if finite guard trips
```

### Phase 9: First Formal Research Run

Only after Phase 8 passes.

Suggested first run:

```text
model_family: date_slate_alpha_fusion_v1
output_profile: forecast_incremental_path_v2
loss_profile: date_grouped_alpha_score_v1
hidden_dim: 128
static fields: exchange,industry
selection_profile: validation_loss
epochs: 3-6
min_epochs: 1
patience: 2
per_epoch_prediction_metrics: disabled
final validation/test prediction: enabled
```

This run is research-only. It does not authorize:

```text
multi-seed
score-backtest bridge
candidate matrix
execution-candidate
paper/live/broker
active/default changes
```

Those remain downstream only if the new single-seed evidence is materially stronger and stable.

## Success Criteria

Code/data contract success:

```text
date-slate pack builds from structured sidecar
loader returns same-date groups with correct labels/static ids
model forward emits incremental outputs and diagnostics
loss computes rank only within date groups
finite guard can stop bad runs with diagnostic payload
checkpoint restore works
```

Training stability success:

```text
no non-finite loss/pred/grad/param in smoke/scout
checkpoint tensors finite
loss components finite and logged
router weights not instantly collapsed
intraday residual norm monitored
```

Research success for first formal run:

```text
validation_loss finite and meaningfully tracked
train/validation loss dynamics interpretable
validation/test prediction outputs available
date-grouped rank diagnostics available
personal topK evaluation can run as external diagnostic
```

## Explicit Non-Goals

Not part of this plan:

```text
changing QDP canonical data
changing QDP active registry pointers
altering daily_research active execution artifact
paper/live/broker integration
promotion gates
score-backtest bridge
candidate matrix
full rolling-window materialization
post-hoc test-only candidate selection as formal evidence
```

## Current Next Action

Implement Phase 1 and Phase 2 first:

```text
1. finite fail-fast diagnostics
2. date-slate sidecar pack builder and loader smoke
```

Do not launch another long h192 no-stride full run before this new contract is implemented or explicitly abandoned by the user.

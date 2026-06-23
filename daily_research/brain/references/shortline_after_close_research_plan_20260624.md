# Shortline After-Close Research Plan 20260624

## Verdict

- Status: `planned / research_only / shadow_only / no_active_change`.
- Date: `2026-06-24`.
- Scope: redefine the next research line as an after-close shortline stock-selection program.
- Verdict: current priority is mechanism-level after-close shortline stock selection, not another path20/deep-model tuning loop.
- Active artifact impact: unchanged. Do not change `daily_research/output/active_execution_strategy.json`, live/default, paper/live/broker, score-backtest bridge, candidate matrix, execution-candidate state, or QDP active registry pointers from this plan.

## User Workflow

The practical workflow is:

```text
D close: update QDP data
D after close: select stocks worth buying tomorrow
D+1: buy only if entry conditions are satisfied
D+2 or later: earliest normal sell window because of A-share T+1
```

Therefore the core research question is no longer a generic 20-day path forecast. It is:

```text
Can after-close technical and market-state information select topK stocks with positive net expectation over short executable holding windows?
```

## Profit Definition

Shortline profit must be treated as positive expectation, not raw prediction accuracy:

```text
net_expectation =
  win_rate * average_win
  - loss_rate * average_loss
  - explicit_cost
  - slippage
  - T+1 locked-in adverse movement
  - tail_loss
```

The model or scorer should therefore search for situations where:

```text
future short-horizon return distribution has a fatter right tail;
left-tail and locked-in risk are controlled;
entry price is not overpaid;
the stock is actually buyable and later sellable.
```

## Initial Profit Mechanisms

Start with three technical mechanisms. Do not mix every market behavior into one large model before these are separately tested.

### 1. Strong Start / Continuation

Hypothesis:

```text
Capital has already entered today, close remains strong, industry/market context supports continuation,
so the stock has higher 1-3 day follow-through probability.
```

Candidate feature families:

```text
raw_limit_up_like_1d
raw_amount_ratio_5_20
raw_intraday_range_1d
intraday_close_position
intraday_last_30m_ret
intraday_close_to_vwap
intraday_price_above_vwap_share
intraday_cum_vwap_slope
intraday_close_pressure_30m
market_limit_up_like_share
industry_ret_5_excess
industry_positive_share_5
stock_ret_5_minus_industry
```

### 2. Strong Trend Pullback / Buy the Dip

Hypothesis:

```text
The stock is still in a short or medium trend, but has pulled back with acceptable intraday support,
so future 1-5 day repair has positive expectation.
```

Candidate feature families:

```text
ret_5 / ret_20
cs_rank_ret_20d
stock_ret_20_minus_industry
intraday_low_to_close_ret
intraday_intraday_max_drawdown
intraday_close_position
intraday_last_30m_ret
amount expansion or contraction
industry_ret_5_excess
```

### 3. Industry Spread / Catch-Up

Hypothesis:

```text
Industry or theme strength is already visible; leaders moved first; selected laggards begin to confirm but are not fully overheated.
```

Candidate feature families:

```text
industry_ret_5_excess
industry_rank_ret_5
industry_positive_share_5
industry_rank_turn
stock_ret_5_minus_industry
industry_member_count
market_breadth_20
market_amount_expansion_share
```

## Lower Priority Mechanisms

Pure oversold reversal and pure limit-up relay are not rejected, but they are not first priority.

Reason:

```text
pure oversold reversal needs strict sell/risk rules;
limit-up relay needs better auction, order-book, seal quality, open-board, and intraday execution information;
existing QDP daily + 5m-derived summary can diagnose them, but first-stage modeling should avoid overclaiming.
```

## T+1 Execution Semantics

Do not use this as an executable return:

```text
D+1 open buy -> D+1 close sell
```

Because A-share T+1 means a stock bought on D+1 normally cannot be sold on D+1.

Existing QDP `next_open` path20 labels are usable for the first fixed-entry baseline because code semantics are:

```text
horizon=1: D+1 open entry -> D+2 open exit
horizon=3: D+1 open entry -> D+4 open exit
```

This is a reasonable first executable baseline.

## Entry Semantics

Do not permanently assume unconditional next-open buying.

Separate the decision into:

```text
selection: is this stock worth watching tomorrow?
entry: at what price and under what condition is buying still positive expectation?
```

Progression:

```text
Stage 1: fixed D+1 open entry baseline.
Stage 2: opening filter, such as high-open no chase, low-open breakdown no buy, limit-up blocked.
Stage 3: entry-zone fill simulation, such as buy only if D+1 price touches an acceptable zone.
Stage 4: 5m confirmation entry, such as VWAP hold, pullback support, or early high breakout.
```

## Current Data Judgment

Current data lake and QDP artifacts are sufficient for first-stage shortline research.

Current QDP facts:

```text
active/latest sharded memmap:
  mainboard_style_structural_alpha_v2_label_v2_full_2010_2026_20260616_01
  feature_profile: style_structural_alpha_v2
  feature_count: 307
  label_schema: path20_basic_v2
  execution_mode: next_open
  years: 2010-2026

full training pack:
  mainboard_style_structural_alpha_v2_label_v2_pack_full_2012_2025_20260616_01
  train: 2012-2023
  validation: 2024
  test: 2025
  sample_count: 7,421,714
```

Useful existing labels:

```text
daily_return
daily_excess_return
cumulative_return_1to20
cumulative_excess_return_1to20
rank_1to20
industry_rank_by_horizon
drawdown_by_horizon
worst_by_horizon
upside_by_horizon
entry_tradeable
entry_limit_up_buy_blocked
entry_suspended_or_no_open
forward_tradeable_ratio_by_horizon
```

Conclusion:

```text
Do not rebuild the large QDP sharded memmap first.
Do not rebuild the full alpha_v2 pack first.
Use the existing pack for Stage 1 fixed next-open shortline diagnostics.
Only derive a new shortline execution label pack when conditional entry is needed.
```

## Needed Derived Label Pack Later

For conditional price-zone entry, existing fixed-entry labels are insufficient. Add a derived pack, not a full QDP rebuild:

```text
shortline_entry_label_v1
```

Potential labels:

```text
next_day_open_gap
next_day_high
next_day_low
next_day_close
entry_zone_fill_flag
entry_fill_price
entry_fill_time_bucket
fill_to_d2_open_return
fill_to_d2_close_return
fill_to_d3_open_return
mae_after_fill_before_sellable
mfe_after_fill
open_too_high_no_chase
low_breakdown_no_buy
limit_up_buy_blocked
```

## Stage Plan

### Stage 0: Contract and Diagnostics Tooling

Goal: create a research-only shortline diagnostic layer over existing alpha_v2 pack.

Required capabilities:

```text
fixed next-open entry simulation;
T+1-aware exits: D+2 open, D+2 close if available later, D+3 open, D+5 open;
topK: 1,3,5,10;
cost model: at least 20 bps round trip, optionally stress at 30-50 bps;
filters: entry_tradeable, entry_limit_up_buy_blocked, entry_suspended_or_no_open;
monthly stability and worst-month reporting;
mechanism bucket reporting: continuation, pullback, industry spread.
```

Stage 0 should not train a deep model.

### Stage 1: Rule / Scorer Baseline

Goal: verify whether each technical mechanism has positive expectation before model complexity.

Build simple, explicit scorers:

```text
strong_continuation_score
trend_pullback_score
industry_spread_score
combined_shortline_score
```

Evaluate:

```text
D close signal -> D+1 open entry -> D+2/D+3/D+5 open exit;
top1/top3/top5/top10;
absolute return and excess return;
net return after cost;
hit rate;
positive month rate;
worst day/week/month;
entry blocked rate;
industry concentration;
turnover proxy.
```

### Stage 2: Opening Filter

Goal: make fixed next-open entry less naive without requiring full 5m execution simulation.

Candidate filters:

```text
do not chase if D+1 open_gap too high;
do not buy if D+1 open breaks the technical setup;
skip entry_limit_up_buy_blocked;
skip suspended or no-open;
optionally require market/industry state not broken at open if available.
```

This stage needs open-aligned diagnostics, but can still use existing next-open labels plus future open/limit labels.

### Stage 3: Shortline Entry Label Pack

Goal: support price-zone entry simulation.

Do not rebuild base QDP. Derive `shortline_entry_label_v1` from existing daily and 5m data.

Primary questions:

```text
Was the desired entry zone touched on D+1?
What fill price should be assumed?
What is the adverse excursion before earliest sellable window?
What is the return to D+2 open / D+2 close / D+3 open?
```

### Stage 4: Lightweight Model

Only after a mechanism shows positive expectation:

```text
train LightGBM/CatBoost or a small neural ranker;
target: same-date shortline alpha score;
horizons: executable 1/2/3/5 day windows, with horizon=1 meaning D+1 open -> D+2 open;
loss: rank/listwise or topK proxy over same-date candidates;
features: mechanism-aligned daily/intraday/market/industry fields;
model role: score candidates, not execute trades.
```

### Stage 5: Conditional Entry Model

Only after Stage 3 labels exist:

```text
predict alpha_score;
predict acceptable entry zone or no-trade;
predict locked-in MAE risk before sellable;
evaluate fill-conditioned returns separately from unconditional returns.
```

## Evaluation Rules

Do not use test-only selection as formal evidence.

Recommended split policy:

```text
train/research fit: 2012-2022 or 2012-2023 depending on task;
validation: 2023/2024 rolling or explicit 2024;
test: 2025;
future live shadow: only after research evidence is stable.
```

Report every candidate with:

```text
selected rule or model;
selection year and target;
same-candidate validation/test performance;
monthly table;
blocked-entry rate;
worst-month attribution;
whether improvement came from selection, entry filter, or exit horizon.
```

## Hard Boundaries

```text
research_only / shadow_only;
no active/default/live/paper/broker changes;
no QDP active registry pointer changes;
no full memmap rebuild before Stage 1 evidence says the current substrate is insufficient;
no deep model before mechanism-level rule/scorer diagnostics;
no D+1 open -> D+1 close executable-return claim under T+1.
```

## Next Actions

- Implement Stage 0 tooling: shortline fixed-next-open diagnostic over existing alpha_v2 pack.
- Compare strong continuation, trend pullback, industry spread, and combined score.
- Report top1/top3/top5, D+2/D+3/D+5 open exits, cost-adjusted net return, monthly stability, and entry blocked rate.
- Do not launch model training before mechanism-level scorer diagnostics are available.
- Do not claim D+1 open to D+1 close as an executable return under A-share T+1.

```text
shortline fixed-next-open diagnostic over existing alpha_v2 pack.
```

Minimum useful output:

```text
one report comparing strong continuation, trend pullback, industry spread, and combined score;
top1/top3/top5;
D+2/D+3/D+5 open exits;
cost-adjusted net return;
monthly stability;
entry blocked rate;
no model training.
```

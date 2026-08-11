# QVER expanding risk-overlay validation

Date: 2026-08-11

## Question

Can the stronger frozen `30/30/25/10/5` interpretable policy be improved by
keeping its exact account Top30 scan universe and selecting the final Top10
with three independently predicted risks: revision reversal, valuation-multiple
compression and early-spike/fade?

## Frozen challenger

The old score and `eligible__exact_full_top10` gate first form the exact
industry-capped Top30 scan list, with at most two names per PIT industry. This
identity set is fixed before outcome reads. The challenger never expands that
candidate set.

Each risk head is a fixed L2 logistic regression with training-median
imputation, missing indicators and robust scaling. Training expands through
time, and a row can enter training only when its legal D120 exit date index is
strictly earlier than the current signal date. At least 360 labelled rows and
30 observations in each class are required. Until all three heads are ready,
the old core order is preserved exactly.

When all heads are ready, their predicted probabilities are converted to
within-date percentile risks and averaged equally. The Top30 is sorted by this
mean risk, with old rank and candidate id used only as deterministic ties. No
core-score blend, hard threshold, risk-head weight grid, band-size grid or
post-result redefinition is permitted.

The matched account freezes gross exposure at `0.6860894024612805`, the old
account's development-only risk budget. All execution, cost, capacity, T+1,
suspension, limit, lot-size, cash and D60 exit rules remain unchanged.

## Audit

- The pre-outcome candidate band contains 4,982 rows across all 168 months;
  monthly size is 6-30 and the two-name industry cap always holds.
- All three heads are simultaneously active in 149 months. Every training
  label has availability strictly before its prediction date.
- The old matched account is reproduced exactly at CNY2,862,788.27, proving
  that candidate identity, order and account execution match the authority.
- All outcomes end no later than 2025-12-31 and the forbidden 2026-read count
  is zero.

## Do the risk heads predict their events?

Yes, weakly but consistently. These are expanding historical OOF predictions
on the frozen Top30 population:

| Head | OOF observations | Full AUC | 2012-19 AUC | 2020-22 AUC | 2023-25 AUC | Brier skill | Lowest/highest risk-quintile event rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Revision reversal | 3,871 | 0.6134 | 0.5784 | 0.6656 | 0.6096 | 3.69% | 32.34% / 58.40% |
| Multiple compression | 4,105 | 0.6067 | 0.5694 | 0.6585 | 0.6428 | 3.21% | 40.70% / 63.49% |
| Early spike/fade | 4,197 | 0.5872 | 0.5968 | 0.5765 | 0.5609 | 1.82% | 17.83% / 26.40% |

The labels also have economic meaning within active months:

| State | Mean D120 when absent | Mean D120 when present |
|---|---:|---:|
| Revision reversal | +13.65% | +0.64% |
| Multiple compression | +22.69% | -7.99% |
| Early spike/fade | +10.38% | -6.62% |

Thus the experiment did not fail because the events are irrelevant or because
the models have no ordering information.

## Selection result

| Policy | Mean D60 | Mean D120 | Early spike/fade | Persistent loss | Revision reversal | Multiple compression |
|---|---:|---:|---:|---:|---:|---:|
| Old core Top10 | 4.39% | 7.92% | 20.29% | 17.24% | 45.54% | 52.58% |
| Risk-overlay Top10 | 3.59% | 6.51% | 17.96% | 17.84% | 50.71% | 49.78% |

Only 794 of 1,676 decisions overlap, or 47.37%. The risk-only rerank is
therefore not economically a small change even though the Top30 universe is
fixed.

The 882 old-only decisions have mean core-band rank 5.70 and D120 return
8.03%. The 882 challenger-only decisions have mean old rank 20.24 and D120
return 5.34%. They reduce early fade from 21.81% to 17.32% and multiple
compression from 52.78% to 47.43%, but revision reversal rises from 42.44% to
52.26%.

The mechanism is conflict among the heads plus loss of core return
information. Revision-reversal and multiple-compression risk ranks have
Spearman correlation `-0.574`; equal averaging lets improvement in one offset
deterioration in the other. Early-fade rank has correlation `0.765` with the
composite, so it dominates much of the final ordering. Selecting almost solely
on event risk also replaces high-core names with much lower-core names.

## Cohort economics

At D120, using the same legal net-return and eligible-pool definitions:

| Policy | Mean monthly Top10 | Excess vs eligible | Excess HAC lower | Positive years |
|---|---:|---:|---:|---:|
| Old core | 7.87% | 2.96% | +0.99% | 12/14 |
| Risk overlay | 6.45% | 1.54% | -0.16% | 12/14 |

The paired challenger-minus-baseline D120 mean is `-1.42` percentage points.
Its HAC 95% interval is `[-2.78, -0.05]` points, so the deterioration is not
just a noisy point estimate under the frozen inference rule. Period means are
negative in every consumed regime: `-1.11` points in 2012-19, `-1.69` in
2020-22 and `-2.08` in 2023-25.

## Exact account result

| Matched 68.61% gross account | Old core | Risk overlay |
|---|---:|---:|
| Liquidated ending equity | CNY2.863m | CNY2.647m |
| Annualized log growth | 8.02% | 7.42% |
| Maximum drawdown | -29.22% | -28.72% |
| Winning-trade rate | 53.44% | 52.58% |
| Positive years | 11/14 | 12/14 |
| Top-10 share of positive P&L | 9.09% | 9.75% |

The challenger improves drawdown by only about 0.50 percentage points while
losing about 0.60 points of annualized log growth and increasing winner
concentration. At full exposure it similarly ends at CNY3.817m versus
CNY4.305m, with drawdown improving only from -40.95% to -40.26%.

## Decision

The frozen challenger fails four economic aims and is **historically
rejected**. It passes only these intended effects: every risk head has AUC
above 0.5, early-spike/fade incidence falls, and maximum drawdown is slightly
smaller. It fails the paired D120 gate, matched account growth gate and winner-
concentration gate; the full acceptance decision is false.

Do not tune head weights, candidate-band width, probability thresholds or
core/risk blend weights on this consumed result. The three heads should be
retained as useful scenario/falsification diagnostics and possible auxiliary
targets, not used as a risk-only final sorter.

The next defensible architecture, if pursued, is a separately frozen
return/downside multi-task action-value model that learns the trade-off between
expected return and the three adverse states inside the existing five-fold and
exact-account contract. It should compete once against the old core rather
than reopening manual overlays.

No live, production or stable-profit claim is allowed.

## Artifacts

- Frozen study: `daily_research/studies/seq100_qver_risk_overlay_validation_v1.json`
- Implementation: `daily_research/path_policy/seq100_qver_risk_overlay_validation.py`
- Authoritative summary: `daily_research/output/path_policy/studies/seq100_qver_risk_overlay_validation_v1/summary.json`
- Candidate OOF predictions: sibling `candidate_band_predictions.parquet`
- Selection comparison: sibling `selection_members.parquet` and `selection_summary.parquet`
- Monthly cohort evidence: sibling `monthly_cohorts.parquet`
- Risk-head metrics: sibling `risk_head_metrics.parquet`
- Exact baseline/challenger account files: sibling `account_*` Parquet files

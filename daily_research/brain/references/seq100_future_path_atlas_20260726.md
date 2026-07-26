# Seq100 pre-2023 future-path atlas - 2026-07-26

## Status

The outcome-space discovery pass is complete. It maps every eligible pre-2023
candidate into an execution-aware D1-D60 future path and discovers path
archetypes without using signal-day predictors. It is descriptive evidence, not
a predictability result and not a successor model contract.

No target, horizon, slot count, leverage, stop rule, loss function, or output
topology is selected. In particular, the result does not establish that a
vector label is better than a scalar label. It establishes only that a single
terminal return loses economically relevant path-shape information.

## Scientific boundary

- Read-only source: the complete-PIT main-board pack
  `daily_research/data/research_store/seq100_pit_l35v2_v1/`.
- Full candidate universe from signal date `2010-09-29` through `2022-09-30`:
  6,096,195 candidate rows over 3,419 security coordinates.
- Every path is anchored at the legal next-day open. The final D60 close is no
  later than `2022-12-30`.
- 2023, 2024, 2025, and 2026 contribute no signal, price, execution, fitting,
  cluster-selection, naming, or stability row to this atlas.
- 2023-2025 remain burned discovery years and may be used only in the separately
  planned three-fold predictability audit. 2026 remains untouched confirmation
  material.
- This work created no study contract, trained no predictive model, and wrote no
  protected pack, QDP dataset, checkpoint, or model-registry entry.

D60 is the pack's observation window, not a selected holding period. The atlas
records intermediate outcomes and legal exits across that window; it does not
assert that a position should be held for 60 days.

## Recorded outcome map

The candidate map retains all 6,096,195 rows. A path is cluster-assignable only
when market entry filled, the next-open anchor is valid, and the D1-D60 OHLC
path is complete. There are 5,929,931 such rows (97.27%). Unfilled and incomplete
rows remain present with explicit status rather than being silently filtered.

For every row the output contains:

- the full 60-value close path relative to the legal next-day open;
- 11 close anchors at D1, D2, D3, D5, D10, D15, D20, D30, D40, D50, and D60;
- MFE, MAE, peak/trough timing, pre-peak adversity, drawdown, post-peak fade,
  persistence, trend slope/R2, efficiency, and first +5%/+10%/+20% hit days;
- observed and sellable fractions, first legal sell day, maximum unsellable run,
  and suspension fraction; and
- costed net return plus actual legal exit day for planned D2, D3, D5, D10,
  D15, D20, D30, D40, D50, and D60 exits.

This is 52 float descriptors plus explicit status bits. A planned exit that
cannot be legally resolved inside the D80 execution window is right-censored to
`NaN`; it is not assigned a zero recovery value.

## Autonomous clustering method

The macro atlas clusters the raw D1-D60 entry-relative close path, without
predeclared archetype names:

1. Cap each calendar year's selection sample at 25,000 rows so later, larger
   universes do not dominate K selection.
2. Winsorize each day at 0.5%/99.5%, center by the median, and scale by the
   10%-90% range.
3. Fit incremental PCA on every assignable row and retain the smallest dimension
   reaching 95% explained variance.
4. Evaluate K=3 through K=10 using equal-weight ranks of silhouette,
   Calinski-Harabasz, Davies-Bouldin, seed ARI, and early-versus-late assignment
   ARI. Exclude clusters below 1% when possible.
5. Refit MiniBatchKMeans on all assignable paths. Repeat the complete selection
   through expanding 2018, 2020, 2021, and 2022 windows.

All four windows independently select K=3 and eight PCA dimensions. Explained
variance ranges from 95.41% to 95.65%. Across the six window pairs, assignment
ARI is 0.734652 minimum and 0.845183 mean; matched centroid correlation is
0.977011 minimum. K=4 through K=10 never wins the combined criterion.

The prototypes are stable, but their prevalence is regime-dependent. Annual
within-universe shares vary materially, so the full-history class proportions
must not be treated as fixed priors.

## Macro outcome map

Percentages and medians below use the 5,929,931 assignable paths. Cluster IDs are
mechanical and have no ordinal meaning.

| macro | descriptive name | rows | share | D5 median | D20 median | D60 median | MFE60 median | MAE60 median | peak day median |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M0 | declining path | 2,270,431 | 38.29% | -2.41% | -7.97% | -12.33% | +4.83% | -21.26% | D6 |
| M1 | strong advancing path | 777,402 | 13.11% | +3.97% | +16.51% | +30.88% | +52.80% | -2.72% | D45 |
| M2 | mild advance / oscillation | 2,882,098 | 48.60% | +1.01% | +2.87% | +3.22% | +18.12% | -8.24% | D34 |

The corresponding median costed net returns are -8.34%/+15.80%/+2.38% at D20
and -13.07%/+28.88%/+2.15% at D60. Those returns are diagnostics for a CNY
100,000 reference order, not an account simulation or a slot allocation.

## Shape atlas inside each macro state

Macro K=3 is dominated by direction and amplitude. To expose path order, each
path is therefore residualized by subtracting the straight line from zero to its
own D60 endpoint. K=2 through K=8 is then selected independently inside each
macro cluster using the same multi-metric rule. Every macro cluster selects K=2,
producing six hierarchical shapes:

| shape | descriptive name | confidence | rows | share | D20 median | D30 median | D60 median | peak day median |
|---|---|---|---:|---:|---:|---:|---:|---:|
| S0 | deep decline then partial repair | tentative | 905,961 | 15.28% | -12.13% | -14.65% | -7.81% | D4 |
| S1 | persistent decline | tentative | 1,364,470 | 23.01% | -5.74% | -7.68% | -14.08% | D8 |
| S2 | progressively accelerating advance | stable | 355,284 | 5.99% | +11.64% | +18.56% | +43.20% | D55 |
| S3 | fast advance followed by fade | stable | 422,118 | 7.12% | +21.23% | +27.18% | +20.18% | D35 |
| S4 | early advance then loss of trend | tentative | 1,486,978 | 25.08% | +5.67% | +6.03% | -1.79% | D26 |
| S5 | late start | tentative | 1,395,120 | 23.53% | +0.25% | +0.87% | +9.08% | D49 |

`stable` requires seed ARI, early/late ARI, and every expanding-prefix ARI to be
at least 0.60. The strong-advance split passes with seed ARI 0.803, early/late
ARI 0.627, and minimum prefix ARI 0.695. The declining split is tentative
because early/late ARI is 0.301. The mild/oscillating split is tentative because
early/late ARI is 0.530 and minimum prefix ARI is 0.483.

The names are post-cluster descriptions of median paths. They are not labels
that the input features have yet been shown able to predict.

## What this says about model outputs

The macro map can largely be ordered by one terminal return, but the shape map
shows why that is incomplete. The same strong-advance macro state contains both
S2, whose median continues to accelerate through D60, and S3, whose median is
already fading after a D35 peak. A terminal scalar collapses that distinction.
Likewise, the mild macro state contains both an early rise that fails and a late
start that has barely moved by D20.

That is evidence for testing path-aware outputs, not for choosing them now. A
vector or hierarchical target carries more estimation error and can be worse if
the shape distinctions are not predictable from signal-day information. A
scalar endpoint can still be the correct output if it is materially more
predictable and matches the eventual account objective. The later
predictability audit must compare, on identical folds and eligibility:

- one scalar endpoint baseline at each candidate horizon;
- macro-class probability or ordinal direction/amplitude targets;
- conditional shape probabilities; and
- a small multi-task descriptor set rather than an unconstrained 60-vector.

No candidate above is frozen. The audit should first ask whether signal-day
inputs contain stable information about the distinctions, then measure whether
that information improves finite-capital decisions. Oracle separability or a
visually intuitive archetype is not learnability.

This also separates entry and later management cleanly. An entry model may
estimate the probability and expected magnitude of a favorable path. A later
state update can react when the realized path moves from the expected region to
another one. The atlas describes those future states; it does not preselect the
exit trigger.

## Reference-order affordability correction

The market-entry mask and reference-order sizing answer different questions.
An audit found 766 rows where the exchange-level entry condition was valid but
CNY 100,000 could not fund one 100-share board lot after slippage and fees. All
766 are `600519.SH`; next-open price is CNY 1,000.00 minimum, CNY 1,763.55
median, and CNY 2,587.98 maximum.

These rows now carry the independent
`reference_order_unaffordable` status bit/column. They remain valid path rows and
remain clustered; only their reference-order net returns are missing. The
repaired map has exactly 766 such rows, zero status/column mismatches, zero
unaffordable rows with realized returns, zero overlap with right-censoring, and
zero unexplained missing return cells.

## Limits and next valid step

- The atlas sees outcomes only through D60. Longer path structures may exist.
- Daily bars cannot represent intraday order, queue priority, or partial fills.
- Cluster selection is unsupervised outcome discovery; no signal-day feature is
  tested here.
- Shape names and all numeric boundaries are discovery artifacts, not immutable
  target definitions.
- No account cash path, overlapping-position queue, or slot interaction is
  evaluated.

The requested three-step path-map work stops here. The next separate task, when
authorized, is the strict 2023/2024/2025 three-fold predictability audit. It must
compare scalar and path-aware candidates without touching 2026 and without
turning these descriptive clusters into a successor contract before they earn
out-of-sample predictive evidence.

## Evidence bindings

- Pack manifest SHA-256:
  `4f8417c2c382c12b9a4055596c572d39988a5d47e91e30736e313da84634fafd`
- Feature-view manifest SHA-256:
  `460a17e0e4067e0356c023f6d3877e8d964b266752a0312bbee6fb1043619617`
- Candidate-index SHA-256:
  `9989c88b56dddca2ea7f380ee7cebf4cc7d56a4d2b5df654fcd5ef2d699b1d03`
- Script `tmp/seq100_future_path_atlas.py`:
  `cacb537d06b2192a14786e5dd424bcd9444a92f258bce006725187ce9867c844`
- Source audit / atlas summary:
  `4d1b32335fd4d598a68ad7c8c251e20705ddf77a2afb46016257ebd7ac3e7516` /
  `ca0747c13a13308b8ccb37b92f0360c0f5a94c213534250eebc91c1958eb36c5`
- D1-D60 path matrix / 52-descriptor matrix:
  `04f615a139d9cd2361d29e2191f745bbdb83633d1bcb1911734b4eb14dcf3718` /
  `c596d5717f307e3eb2644e914342bf3a464d714d9d1fff4ece13c38e1856814f`
- Status / macro-cluster arrays:
  `d669c230f0a2249604d43a7bd87faa18e944b06271a9afbaa06dfbe1ccad27c5` /
  `e073d1d92133900c66b06622d674d62c32b90734e25a138712fa9e1d167b288f`
- Candidate path map / macro summary:
  `25e02036df79e80cb760fd5514f243b5cdd64041129a8e42b84b07c52de5d9a9` /
  `9204acda565ae3ad37d93aaf07ee6b1672d85e2b2c0bbc7ef665475a39d73368`
- Shape atlas summary / shape-cluster summary / candidate shape map:
  `17bc61e66c408970aad518cfb9b6fac23f3058a4d5a4568cf81cf9058a491da0` /
  `916b0f1936d884c5e78b79c8cb191f4ab591741e14f5a719bb5299bf4be7e8ef` /
  `e3fce05402a88f799cf86a7e7485c9cffb0d6c4d57f1c89175d29d4c3b99748a`

The script and generated process material remain ignored under `tmp/`. The
Brain reference is the durable conclusion; generated output is retained for
reproduction but is not a protected truth source.

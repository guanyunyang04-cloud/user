# Seq100 pre-2023 future-horizon atlas - 2026-07-26

## Status

The requested D5/D10/D20/D40/D60 clustering sensitivity test is complete. It
shows that the apparent number of future-path states and the meaning of a
"strong" state depend materially on how much future path is visible.

This is an outcome-space audit only. It does not test whether any state can be
predicted from information available before entry, and it does not select a
target, holding period, exit rule, slot count, leverage, or successor contract.

## Scientific boundary

- Read-only source: the completed pre-2023 future-path atlas under
  `tmp/seq100_future_path_atlas/attempt_001/`.
- Full candidate universe: 6,096,195 rows from signal date `2010-09-29` through
  `2022-09-30`; 5,929,931 rows have a legal next-open anchor and complete D1-D60
  path and are cluster-assignable.
- The latest D60 outcome is `2022-12-30`. No 2023, 2024, 2025, or 2026 signal,
  price, fitting, K-selection, naming, or stability row is used.
- D5 through D60 are observation prefixes, not selected holding periods.
- No signal-day predictor feature is loaded. No protected pack, QDP dataset,
  checkpoint, model registry, or active study contract is changed.

## Method

Each horizon independently clusters the raw entry-relative close prefix D1-DH:

1. Use the same per-year balanced sampling, 0.5%/99.5% clipping, median
   centering, and 10%-90% range scaling as the source atlas.
2. Fit incremental PCA on every assignable row and retain the smallest number
   of components reaching 95% explained variance.
3. Search K=2 through K=10 using the equal-weight ranks of silhouette,
   Calinski-Harabasz, Davies-Bouldin, seed ARI, and early-versus-late assignment
   ARI. Clusters below 1% are ineligible when another eligible solution exists.
4. Repeat the full selection through expanding 2018, 2020, 2021, and 2022
   windows, then refit on every assignable row in that window.
5. Keep a fixed-K3 companion fit at every horizon so low/middle/high states can
   be compared without changing the number of states. These names are assigned
   after fitting by median return at the clustering horizon; they are not model
   labels known before entry.

The final 2022-window PCA dimensions are 3/5/6/7/8 for
D5/D10/D20/D40/D60, explaining 95.71%/96.60%/95.62%/95.19%/95.41%.

## Result 1: the selected resolution depends on horizon

| prefix | through 2018 | through 2020 | through 2021 | through 2022 |
|---|---:|---:|---:|---:|
| D5 | 3 | 3 | 2 | 3 |
| D10 | 2 | 3 | 2 | 2 |
| D20 | 2 | 2 | 2 | 2 |
| D40 | 2 | 2 | 2 | 2 |
| D60 | 2 | 2 | 2 | 2 |

When K=2 is allowed, the combined criterion consistently prefers a coarse
low/high split from D20 onward. The source D60 atlas selected K=3 because its
declared search range was K=3 through K=10. That result remains correct within
its contract, but K=3 is a useful analysis resolution rather than evidence that
the outcome space has exactly three natural classes.

D5 and D10 sit on a binary-versus-ternary boundary. At D10 through 2020, K=3
wins by only 0.2 mean-rank points over K=2 (2.4 versus 2.6). K=3 has better
early/late assignment ARI (0.541 versus 0.246), while K=2 has better silhouette
(0.334 versus 0.322). The other three D10 windows select K=2. This is a real
resolution ambiguity, not a stable third economic state.

## Result 2: long prefixes are more calendar-stable

Minimum assignment ARI across the six expanding-window pairs:

| prefix | autonomously selected K | fixed K3 |
|---|---:|---:|
| D5 | 0.178 | 0.790 |
| D10 | 0.191 | 0.642 |
| D20 | 0.859 | 0.662 |
| D40 | 0.903 | 0.645 |
| D60 | 0.920 | 0.749 |

The very low D5/D10 autonomous ARI values occur when K changes between 2 and 3.
With fixed K3, the short-prefix partitions are substantially more stable. From
D20 onward, the autonomous binary partition itself is highly stable across
calendar expansions.

The D10 autonomous matched-centroid correlation minimum of 0.0259 is a metric
artifact under unequal K, not evidence that the path prototypes reversed. The
2020 K3 fit has near-flat, high, and deep-low centers ending at +0.21%, +10.37%,
and -8.59%. The 2021 K2 fit ends at -3.95% and +6.93%. Minimum-RMSE Hungarian
matching pairs the near-flat 2020 center with the 2021 low center and leaves the
deep-low center unmatched; that pair has correlation -0.948 while the high/high
pair has correlation 0.99999, producing the misleading mean. The low ARI still
correctly says the partition resolution changed, but the 0.0259 centroid number
must not be interpreted semantically.

## Result 3: short-term strength is not long-term strength

The fixed-K3 companion gives a common low/middle/high vocabulary. Agreement
below is against the independently fitted source D60 macro atlas. The final
three columns describe the D60 return distribution inside the prefix-high state.

| prefix | ARI to D60 | overall purity | high -> D60 strong M1 | high D60 q10 | high D60 median | high D60 q90 |
|---|---:|---:|---:|---:|---:|---:|
| D5 | 0.071 | 57.06% | 33.13% | -16.77% | +5.17% | +42.91% |
| D10 | 0.118 | 60.61% | 37.44% | -14.47% | +7.21% | +44.98% |
| D20 | 0.271 | 69.92% | 54.60% | -9.57% | +13.16% | +55.66% |
| D40 | 0.622 | 86.79% | 80.19% | -0.46% | +22.97% | +67.69% |
| D60 refit | 0.812 | 93.85% | 85.83% | +7.53% | +28.47% | +68.98% |

A D5 high state is therefore only early strength: 14.39% of those rows later
belong to D60 declining M0, 52.48% to mild/oscillating M2, and 33.13% to strong
advancing M1. By D40, the high state maps to M1 80.19% of the time, while the
low and middle states map to M0 and M2 91.1% and 85.4% of the time.

The same conclusion appears in adjacent-prefix agreement. Fixed-K3 ARI/purity
is 0.459/80.1% for D5->D10, 0.418/77.8% for D10->D20, 0.439/79.1% for
D20->D40, and 0.619/86.6% for D40->D60. State identity becomes materially more
locked only after the path has accumulated roughly 40 observations.

This supports the owner's statement that price paths are "walked out": early
strength raises the chance of a strong continuation but does not determine it.
It does not imply that a trader should wait 40 days to enter. D40 is future
information relative to the original signal and is used here only to measure
how uncertainty resolves.

## How to compare good and bad inside one cluster

Clustering and ranking answer different questions. Distance to a centroid says
how typical a row is for that cluster; it does not say whether the row is more
profitable or economically better. In particular, the most typical D5-high path
need not be the D5-high path most likely to finish in D60 M1.

Within-cluster quality should therefore remain an explicit conditional outcome
profile rather than another hidden cluster ordering:

- terminal amplitude and its q10/median/q90 distribution;
- transition probability into later low/middle/high states;
- MFE, MAE, pre-peak adversity, drawdown, fade, persistence, and time to reach
  return thresholds; and
- entry/exit legality and costed realizability.

For example, D5-high is internally split into a later declining group, a mild
group, and a strong group. Its D60 q10-to-q90 range is -16.77% to +42.91%, so a
single D5-high class label discards most of the economically relevant ordering.
The existing hierarchical shape atlas also shows that even D60 strong M1
contains progressive acceleration and fast-rise-then-fade paths.

No scalar weighting of those dimensions is frozen here. Such a weighting would
silently reintroduce the target-design decision that this audit is meant to
inform. The next predictability audit must determine which distinctions are
actually learnable from signal-day inputs before deciding whether the successor
should emit a scalar, class probabilities, conditional quality, or a small
multi-task vector.

## Validation

- The script passed `py_compile`, the label-invariant synthetic transition
  self-test, and the source-boundary audit.
- Independent full-array validation confirmed exactly 5,929,931 nonnegative
  labels in every horizon/family, all 166,264 invalid candidates equal to `-1`,
  and the expected unique label counts for selected K and fixed K3.
- All 6,096,195 Parquet rows were compared in batches against both memmaps and
  the source D60 macro array. Candidate IDs and all 11 label columns had zero
  mismatches.
- All 138 horizon-transition rows and 78 source-macro transition rows sum to one
  within floating-point error; every comparison covers all 5,929,931 assignable
  rows.
- Parquet row-group statistics independently confirm signal dates
  `2010-09-29` through `2022-09-30`; the bound outcome cutoff is `2022-12-30`.
- All three charts were visually inspected. The heat-map colorbar layout was
  corrected without refitting any model or changing any label or metric.

## Evidence bindings

- Script `tmp/seq100_future_horizon_atlas.py`:
  `caf6e93ba705366098f9c07aaa965491f94ab8047d5111c7b4f6d98dcfc393b9`
- Source audit / horizon summary:
  `5f7d63f055267bb81a3c6f7d483265ad8855f976cd5f21c00263140c3e5df294` /
  `ca6603734ae973ace047d4198293c342e45e777840e2bf1b1d1973b3ef687a5e`
- Selected-K / fixed-K3 label arrays:
  `053d7aa8c3b0f813b7d16b4ab85cb7531acf562d72617962d8851700702eb001` /
  `98420eb75656c83ef80f6bbb708b93d9da335ef0798ebf1e1215ab19ea4e1579`
- Candidate horizon map:
  `4d2640bbb94371526973059ba6280e6b2f0304d57b5b31ecf4b90ee0b1fd016f`
- Selected / fixed-K3 profiles:
  `156c53afe052c3d3d4edbc997b4d9e04e45a6e5b7261c6e4e89b349e9fbe6444` /
  `a1f48bfad9b78d66dca4d7369a23a4f03bd6459e9061a01cd65c4fad0266ada9`
- Calendar stability / horizon-pair metrics / horizon transitions:
  `864e1014cd855bb2b7ba5ca528a1252ffc04543fd680dd2753ad197a566463ab` /
  `99ff294d79cd7691ae656eb92de04b39d777679675ee62374c4fbd1ae0d3e22e` /
  `1b4c57fb151cb5eef0c7c707ced5f89d3d1bd8cffe084f4a13b07393514612a4`
- Source-D60 metrics / transitions:
  `0b53282e55f5593502b00d3f5360da05ede19e95ca82c248f91285ff01ae48b6` /
  `efe90edcd2ac29c23d2136e4b845773d110f1efb78a5a4f184593ca0f5b0fa1e`
- Profile / transition / agreement charts:
  `071fe86a5b522e0bc50bc4564bf2e5191643069c76fde49a46c7a4367463f997` /
  `da4a49ec734b05d83c20ecff30ecd5f5d21117922972edcda94ff00bf24e7abd` /
  `c5251eb56aaca743de727890e4349eb887801e1374f1d01e846fbb9cbd678119`

The script, models, label arrays, maps, and charts remain ignored process
material under `tmp/`. This Brain reference is the durable conclusion and does
not make the generated output a protected truth source.

# Seq100 learnability candidate labels and base inputs - 2026-07-27

## Status

The two owner-authorized preparation steps are complete:

1. Candidate labels were generated independently for D5/D10/D20/D40/D60.
2. One uniform causal F1-F5 feature view was bound by read-only reference.

No predictability model was trained. No candidate label, horizon, target shape,
exit rule, slot count, leverage, stop, successor contract, or 2026 confirmation
was selected.

## Scientific boundary

- The row universe is the complete candidate index with 8,674,588 rows. It is
  not filtered by future entry fill or future label availability.
- There are 8,518,212 signal rows through `2025-12-31`. The 156,376 rows after
  2025 remain aligned but have missing labels, states, flags, and entry outcome.
- Every source read is bounded by the horizon. The maximum outcome date read is
  `2025-12-31`; no 2026 entry, price, path, mask, state fitting, or screening
  value is consumed.
- Late-2025 missingness is horizon-specific. A missing D60 outcome does not
  remove an available D5, D10, D20, or D40 outcome.
- The protected `seq100_pit_l35v2_v1` pack is read-only. The closed
  `seq100_pit_signal_quality_v1` feature view is reused only as architecture
  material. Its old target manifest and D60-purged fold views are not consumed.
- The 2023-2025 years remain burned discovery/evaluation years. The generated
  material can support a later three-fold predictability audit, but this step
  does not perform that audit and does not alter the 2026 freeze.

## Candidate label semantics

All conditional path labels use the actual next-trading-day open as the entry
anchor. They require a known and filled entry, a valid anchor, and a complete
D1-DH OHLC prefix.

| label | definition |
|---|---|
| `g_H` | Gross simple return from the next-open entry to the D_H close. |
| `mfe_H` | Highest entry-relative close return among source-legal close exits in D2-DH. It may be negative and is not floored at zero. |
| `pre_peak_mae_H` | Minimum of zero and the D1-to-best-sellable-close close path; it is always non-positive. |
| `state_H` | Assignment of the D1-DH close path to the through-2022 frozen fixed-K3 centers, reordered by raw center endpoint as low/mid/high. |
| `entry_fill` | Next-open execution outcome over the unfiltered candidate universe: `-1` unknown, `0` not filled, `1` filled. |

`mfe_H` deliberately uses legal closing marks rather than daily highs. This
avoids assuming perfect intraday execution and avoids the unknown ordering of a
same-day high and low. D1 is excluded by the A-share T+1 sell rule. The labels
are gross path descriptions and therefore introduce no account size, lot-cost,
slot, leverage, or terminal-recovery convention.

The fixed K3 state is a taxonomy, not a complete quality rank. The separate
`g_H`, `mfe_H`, and `pre_peak_mae_H` values retain amplitude and adversity
inside each state. The state centers were not refit on 2023-2025.

## Horizon coverage

| horizon | latest complete signal | candidates with complete calendar outcome | valid `g_H` / state | valid `mfe_H` and `pre_peak_mae_H` |
|---|---|---:|---:|---:|
| D5 | 2025-12-24 | 8,502,896 | 8,326,420 | 8,321,235 |
| D10 | 2025-12-17 | 8,487,590 | 8,311,178 | 8,309,453 |
| D20 | 2025-12-03 | 8,456,963 | 8,280,652 | 8,280,570 |
| D40 | 2025-11-05 | 8,395,660 | 8,219,582 | 8,219,582 |
| D60 | 2025-09-30 | 8,334,334 | 8,158,478 | 8,158,478 |

The small D5/D10/D20 difference between endpoint and MFE coverage is the
explicit no-legal-close-exit case, not a zero-recovery substitution. In
2023/2024/2025, known next-open fill rates are 99.834%/99.453%/99.682%.

## Uniform causal input

The base input is a row-aligned read-only view of 296 continuous and five
categorical features:

- F1: 105 raw price, K-line, return, trend, volatility, location, turnover, and
  volume-price features with lookbacks through 60 trading days;
- F2: the 105 same-day cross-sectional percentiles of F1;
- F3: 56 market and index breadth, return, volatility, liquidity, suspension,
  ST, and limit-state features;
- F4: eight continuous and two categorical industry features;
- F5: 22 continuous and three categorical size, valuation, share-capital,
  membership, listing-age, and status features.

Every feature is available at signal-date close or earlier.
`entry_buyable_used_for_features=false` and
`future_labels_used_for_features=false`. A 60-day technical lookback is only an
input history length; it does not privilege D60 as an outcome horizon. New
horizon-aware folds must be created later rather than inheriting the retired
study's 60-day purge contract.

## Validation

- Five synthetic semantic checks passed: T+1 exclusion, an unsellable peak,
  zero-bounded pre-peak adversity, unfilled-row missingness, and raw-cluster to
  ordered-state mapping.
- A full Parquet scan matched all 8,674,588 `candidate_id`, `date_idx`, and
  `symbol_idx` rows to the feature and label coordinates exactly.
- Full-array invariants confirmed valid flag/missingness equivalence, state
  values in `{-1,0,1,2}`, horizon-specific cutoff enforcement, and 156,376
  completely missing post-2025 rows.
- A deterministic 4,112-row source recomputation, including every horizon
  boundary, had zero missingness, flag, state, entry, or numeric mismatch; the
  maximum numeric absolute error was 0.
- An independent full-array cross-check against the pre-2023 atlas covered all
  5,929,931 assignable paths at each horizon. Frozen K3 state mismatches were
  zero and `g_H` maximum absolute error against the old close path was 0.

## Evidence bindings

- Script `tmp/seq100_learnability_inputs.py`:
  `928a7b035b4637d2485771640f0d4c82eb8fe7427b08385a4ee14d9512917ae9`
- Label manifest file / resolved payload:
  `e1094434d6d8a094e56812718ac31c4a2730ac2a3b04cd257368312555936a54` /
  `9e0a9c8d6a00219881f87a9e78105ec3e9d5dbd7ed39c1b4cfabed09859e5489`
- Base feature manifest file / resolved payload:
  `5edb948d8a94d7e6127888ebe1e7fcf4dc60c7aef534c2978f2c9b71eecad2f3` /
  `3d4e405dc6ee0312fe2a2c71ec046d410cb692542e22546eafd9b6ff7519b1a3`
- Candidate labels / state labels / flags / entry fill:
  `3a8d6fdea77049e7e1ff4573c3935c4b5519bf3ca800efa98bacd85dda81568a` /
  `1b190ade66d9e124c9f100c2ec9a3d44f42dcbd88bcefda04a6e59c9642f4ef7` /
  `423807817c95f9e15369d6d87f8c0b2e464e502236cd709cdaf770cd7f96ec9f` /
  `4f71961ea2d05e925f55ba9d30a0ac1ac3c814fbcf25de2665514dff68c5e303`
- Audit: `812d412b54708c55a9cb43f274aa7040d23023eb32312bbc13c65c1aacf2c73e`.
- Source protected-pack manifest:
  `4f8417c2c382c12b9a4055596c572d39988a5d47e91e30736e313da84634fafd`.

The generated arrays, manifests, summaries, script, and run log remain ignored
process material under `tmp/seq100_learnability_inputs/attempt_001/` and
`tmp/seq100_learnability_inputs.py`. This Brain reference is the durable
conclusion and does not register the process output as a formal dataset, model,
study target, or protected truth source.

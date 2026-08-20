# Current project state

Updated: 2026-08-20

## Objective

Evaluate repeatable A-share cross-sectional ranking signals under point-in-time
information timing and legal after-cost execution. Results are probability
evidence, not a promise of profit. No live trading is active and research
never reads outcome data after 2025-12-31.

## Active paths

- Code: `src/quantlab/data` and `src/quantlab/research`.
- QDP data: `data/qdp/qdp_v2/active/active.json`.
- Research contract: `data/research/daily/manifest.json`.
- Model outputs: `runs/daily/`.
- Durable evidence: `research/records/` and `research/studies/`.

The former `daily_research` and `quant_data_platform` trees were removed from
the active repository. Their tracked source history remains in Git. The large
ignored experiment artifacts previously staged at
`H:\quant_project_archive\20260820` were purged on 2026-08-21; only small
audit, migration, and legacy-provenance files remain there.

The unified package passes the full test suite and static checks. The QDP
physical contract checks are green; the deep check reports one expected
medium-coverage warning because historical 5-minute bars are unavailable for
some restored daily stock-days. That gap is explicit in the QDP contract and
is not used as a hidden eligibility filter.

## Data and execution contract

- 4,191,476 unique stock-days, with 2010-2011 used only as warm-up and a
  maximum outcome date of 2025-12-31.
- 158 fields are stable daily price-volume and market-state inputs. 183 adds
  25 same-day 5-minute summaries and remains a challenger, not the default.
- Five expanding forward folds cover 2020-2025 with a 30-day purge.
- The primary target is executable D10 net return ranking. Replay uses next-open
  entry, T+1, suspension/limit restrictions, 100-share lots, finite cash,
  delayed legal exits, fees, and double-slippage stress.
- Missing, unfilled, or unaffordable selections remain cash; they are not
  replaced after looking at future outcomes.

## Established findings

- The corrected 158-field tree and raw-60-day sequence both carry weak positive
  out-of-fold ranking information. Their errors are complementary, so a frozen
  50/50 rank ensemble is the current research benchmark.
- The 183-field view has not shown stable incremental economic value over 158;
  minute fields receive little tree gain and may be more useful for shorter
  execution horizons than for D10 selection.
- The benchmark's raw return still depends materially on a small number of
  right-tail winners. Winner-capped and seed-stability results are the gating
  evidence before any paper-trading candidate is considered.
- Financial, announcement, and revision fields are not default inputs. They
  must be PIT- and age-aware and prove residual value over the technical
  baseline.

## Immediate next action

Rerun the predeclared raw-60 seed check and paired ensemble increment using the
unified package. Do not search
blend weights, TopK, exit grids, minute subsets, or LightGBM parameter grids on
the already-used folds. Build a small historical/live parity path only after
winner-cap and seed-stability checks pass.

# Seq100 PIT Pack Semantics Implementation

Date: `2026-07-11`

Status: `implementation verified / corrected full pack blocked at QDP price-substrate gate / active execution unchanged`

## Implemented research-pack contract

`daily_research/path_policy/qdp_v2_sequence_path_pack.py` now supports an explicit, additive research mode with:

- canonical back-adjusted OHLC (`raw OHLC * adjust_factor`, whose active QDP contract defines `adjust_factor` as the positive back-adjust factor);
- raw next-open prices retained separately for exchange-tick execution checks;
- `open_below_limit_tick`: an order fills only when the raw open rounded to `0.01` CNY is below the rounded up-limit and the security is not suspended;
- signal-day PIT eligibility sourced from an immutable `pit_signal_universe` manifest, without requiring activation of that research-scope pointer;
- unfilled next-open candidates retained in `sample_index.parquet` with an `entry_filled` field;
- explicit suspension valuation: carry the last adjusted close, set volume/amount to zero, and keep tradability false;
- separate `price_label_valid` and `va_aux_valid` masks;
- observation, status, tradability, previous-close, correlation-validity and zero-range masks;
- non-materialized `observed_price_path` and `tradable_path` future-window views over the two-dimensional masks;
- manifest provenance for price, entry, sample-filter and suspension semantics.

Legacy defaults remain `none / legacy_999_tolerance / complete_case_filled / none`; existing artifacts are not overwritten.

## Verification

- Focused pack tests: `47 passed`.
- Mainline compatibility tests: `17 passed`.
- Walk-forward compatibility tests: `11 passed`.
- A real-QDP short smoke pack completed and validated before the canonical PIT universe was connected: 20,822 samples over seven signal dates, including 60 unfilled candidates retained in the sample index; manifest file-size validation returned `ok`. The temporary smoke artifact was removed after verification.
- `git diff --check` passed.

## Gate-0 blocker discovered

The immutable PIT universe is complete enough to expose the survivor mismatch, but the active `market_daily_raw` and `adjust_factor` substrates still use the current-survivor 3037-symbol scope.

An exact anti-join for planned signal dates `2018-01-01..2025-12-31` found:

- `402,987` PIT-eligible symbol-days without an active daily-price row;
- `347` affected symbols;
- first/last affected dates: `2018-01-02..2025-12-31`.

Examples begin with `000005.SZ`, `000010.SZ`, `000016.SZ`, `000018.SZ`, and `000023.SZ` on `2018-01-02`.

The builder now runs this exact anti-join before allocating large pack arrays. A corrected PIT build fails with a machine-readable `progress.json` blocker rather than silently shrinking back to the survivor universe.

## Required next data step

Import and validate a canonical PIT-complete daily OHLCVA substrate for the same historical universe, plus a compatible corporate-action adjustment source. Only after both price and factor coverage align with `pit_signal_universe` may the 2018-2021 inner-fold pack be built and the new model experiments begin.

This implementation does not change any active execution, paper, broker, or promoted model artifact.

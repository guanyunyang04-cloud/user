# Seq100 Training-Core Implementation

Date: `2026-07-11`

Status: `implementation verified / screening not started / active execution unchanged`.

## Implemented model semantics

- Added the explicit `hard_st` path-value gradient profile. Its forward value is the exact hard maximum used by NumPy inference; its backward gradient is the existing temperature-smoothed log-sum-exp gradient. The historical `smooth_current` profile remains the default for old profiles and evidence reproduction.
- Added four-field path diagnostics for `predicted_exit_day`, `opportunity_value`, `realized_plan_value`, and `oracle_regret`.
- Corrected packs can provide future tradability masks. Realized opportunity targets cannot choose a suspended exit; predicted paths never receive the future mask. A predicted suspended exit is deferred to the first later tradable day.
- Signal-day candidates that fail the next-open fill rule remain ranked. Their realized-plan return/value is zero (cash), and entry fill rate and realized-plan coverage are reported.
- Corrected-pack availability masks are appended to model inputs: `has_bar`, `is_suspended`, `previous_close_valid`, `zero_range`, plus `corr_valid` when intraday input is active.
- Price-path MAE, field MAE, entry fill rate, observed-path rate, and tradable-path rate are now split diagnostics.

## Implemented ranking candidate

- Added `global_tail_512` as an opt-in rank-training profile with the frozen `32/128/128/96/128` allocation.
- Path reconstruction uses full-sample mixed-date batches. Every four path steps, the trainer can run one separate single-date rank step; any remaining date slates are drained at epoch end so the interval never drops dates.
- The rank loss is target-gap weighted pairwise logistic loss with additional weight on pairs whose better member is in the true Top32.
- Epoch 1 uses static target strata plus random body negatives. Later epochs take the highest predicted false positives outside the true Top256 from one full-train, `eval`/`no_grad` scoring pass made only after the prior epoch's path and rank updates are complete. Mining time and throughput are persisted in training history.
- The added mining pass costs one extra full-train forward pass between epochs (no backward pass and none after the final epoch). A four-sample CPU fixture measured `0.0023 s` / `1,738 samples/s`; production cost must be read from the persisted GPU history rather than extrapolated from this fixture.
- Each included date contributes one rank slate, so rank dates are equally weighted. The exact daily listwise VJP remains conditional and is not implemented or activated.

New comparison commands:

- `train-daily-only-summary-v2-ohlcva-aux-low-hard-st`
- `train-daily-only-summary-v2-ohlcva-aux-low-hard-st-global-tail`

The existing default command and active research pointer were not changed.

## Training-pipeline changes

- Batch loss diagnostics remain on CUDA and cross to CPU once per epoch instead of ten scalar synchronizations per batch.
- The trainer no longer calls `EmptyWorkingSet` every 200 batches; trimming remains at epoch/split boundaries.
- Normalization arrays are cached and training batches do not construct unused trade-date/symbol strings.
- A one-thread pinned-memory prefetcher overlaps the next memmap/normalization gather with the active CUDA step.
- Mixed-date path batches eliminate date-remainder waste for `global_tail_512`; batch `1024` remains an experimental optimization setting because it changes optimizer-step count.
- Training history now records wall-clock throughput, separate rank-batch counts/loss, and total optimizer steps.

## Real-throughput benchmark

Protocol: existing today-close pack, daily-only input, `gru_ohlcva_aux_path_value`, hidden 128, two GRU layers, main loss weights, RTX 2060, 65,536 training samples.

| Batch | Prefetch | Throughput | CUDA peak |
|---:|---:|---:|---:|
| 512 | 0, first/cold-order run | 4,255 samples/s | 0.594 GB |
| 512 | 1, first run | 6,487 samples/s | 0.594 GB |
| 1024 | 1 | 9,523 samples/s | 1.150 GB |
| 512 | 1, reverse-order warm run | 6,962 samples/s | 0.594 GB |
| 512 | 0, reverse-order warm run | 6,144 samples/s | 0.594 GB |

The warm reverse-order comparison supports about `+13.3%` from prefetch alone. Cold-page benefit is larger but is not treated as a stable effect. Batch 1024 was about `+36.8%` over warm batch-512 prefetch, but still requires an equal-sample learning experiment before adoption.

## Verification and evidence boundary

- The fixed-OOS training entry point now recomputes and verifies the immutable fold-training contract itself, including the purged split and pre-OOS normalization cutoff; direct CLI use cannot bypass walk-forward provenance checks.
- Focused training/pack/mainline/walk-forward regression subset: `92 passed`; the broader combined regression run earlier in this implementation line had `120 passed`.
- Python compilation and `git diff --check` passed. Ruff was unavailable in the `yolos` environment.
- No corrected PIT full pack or model screening run was started. The pack implementation found that active QDP daily-price/factor substrates miss 402,987 PIT-eligible symbol-days across 347 symbols for 2018-2025; that Gate-0 blocker must be resolved first.
- These are implementation and throughput facts, not model-quality, outer-OOS, promotion, or execution evidence.

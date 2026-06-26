# Provider Eval 20260626 01

## Scope

- Run tag: `provider_eval_20260626_01`
- Output root: `H:/quant_project/quant_data_platform/data/provider_eval/provider_eval_20260626_01`
- Providers: `akshare`, `baostock`, `efinance`, `mootdx`, `cninfo`, `current_qdp`
- Symbols: `000001.SZ`, `600000.SH`, `300750.SZ`, `688001.SH`, `000300.SH`
- Windows: 2010-01, 2015-06, 2020-03, 2024-06, 2025-12, 2026-06 sample windows
- This was a read-only data-source evaluation. It did not change canonical, registry, memmap, training pack, or data lake content.

## Main Results

- Overall: 141 endpoint records, 49 ok, 92 errors, 12,073 normalized market rows, 1,464 cross-source OHLCV diff rows.
- `baostock`: 37/37 endpoints ok. Strongest batch historical-data candidate in this probe, but slow: market daily p50 around 71-74 seconds per sampled window, p95 around 101-111 seconds.
- `current_qdp`: 2/2 endpoints ok. QDP baseline read was fast and stable; it remains a comparison baseline, not an absolute truth source.
- `mootdx`: 3/3 endpoints ok. Quote and daily small probes were fast. Returned price fields matched QDP on overlapping rows, but volume needed a 100x factor to align, consistent with hand/share unit semantics.
- `cninfo`: 2/2 endpoints ok for the lightweight announcement probe. This supports using it as an announcement/disclosure capability candidate, not as an OHLCV source.
- `akshare`: import ok, but only 4/60 endpoint records ok in this environment. Most failures were Eastmoney `ProxyError`; returned OHLC prices matched QDP where data was available. Volume needed a 100x factor to align.
- `efinance`: import ok, but only 1/37 endpoint records ok. Failures were mainly Eastmoney `ProxyError` and `JSONDecodeError`; not suitable as a primary data path in the current network environment.

## Cross-Source Semantics

- Price fields (`open/high/low/close`) matched QDP exactly on overlapping unadjusted rows for `akshare`, `baostock`, and `mootdx`.
- `baostock` amount and volume were already aligned with QDP up to tiny numeric differences.
- `akshare` and `mootdx` volume needed `suggested_unit_factor = 100.0`; this should be treated as a mandatory normalization rule before these sources are used for lake ingestion.
- Amount differences for overlapping rows were tiny relative to notional size and consistent with rounding or source precision differences.

## Source Tiering From This Probe

- Primary historical OHLCV candidate: `baostock`, subject to throughput/backfill planning because it is stable but slow.
- Existing canonical baseline: `current_qdp`, usable for local read stability and cross-source comparison but not assumed absolutely correct.
- Minute/realtime/supplement candidate: `mootdx`, promising for fast small probes; requires explicit unit normalization and local-cache/backfill design before production use.
- Disclosure/announcement candidate: `cninfo`, promising for lightweight announcement search; needs a separate disclosure-date/PIT audit before entering a canonical slow-disclosure domain.
- Conditional exploration only: `akshare`, because network reliability was poor in this environment despite semantically good overlapping OHLC rows.
- Not recommended as primary path now: `efinance`, because the current probe showed poor reliability.

## Next Steps

- Do not adjust data-source choice using model training returns.
- For a primary-source decision, run a focused `baostock` throughput/backfill scout on a larger symbol set and multiple years.
- For `mootdx`, run a dedicated daily/minute local-cache probe and formalize volume/amount unit normalization.
- For `cninfo`, build a PIT-safe disclosure-date probe before using it for financial-report or announcement domains.
- Keep `akshare` and `efinance` as optional secondary probes until the Eastmoney proxy/JSON reliability problem is solved or isolated.

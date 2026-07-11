# Seq100 Formal PIT View And Corrected Source Pack

Date: `2026-07-11`

Status: `data and source-pack gates passed / model screening not started / active execution unchanged`.

## Formal QDP Research View

The non-active view is:

`quant_data_platform/data/qdp_v2/views/seq100_pit_2012_2025_formal__9d6feb2a5ba7f15a7635b42f.json`

It atomically overrides market daily, security status, limit status, adjust factor,
and PIT signal universe. The underlying range is `2012-01-04..2026-06-01` so late
2025 signals retain 60 future trading days. Quality evidence is:

- 3,409 symbols and 3,496 dates;
- 9,530,656 market/factor/limit rows;
- 9,542,426 PIT/status rows and 8,805,538 signal-eligible rows;
- zero duplicate keys, missing/nonpositive factors, missing factor provenance, or
  independent research-eligible/non-suspended market gaps;
- security-status and PIT row counts align exactly;
- `active.json` SHA-256 remained
  `E56F72A6CBA8BCF86055817F6A0EC5E7391271FB3C27B4D628C3ABC62944051E`.

Gate 0 no longer checks a derived signal flag that already requires a market row.
It independently anti-joins every research-eligible, non-suspended key to the
valid market table and verifies the expected market start/end coverage.

## Corrected Source Pack

Manifest:

`daily_research/data/research_store/sequence_pack/qdp_v2_seq100_path60_todayclose_pit_adjusted_2012_2025_v1/manifest.json`

The pack uses 100 lookback days, 60 forward days, today-close anchoring,
back-adjusted OHLC, raw tick-rounded next-open entry checks, PIT signal eligibility,
carried adjusted close for suspension valuation, and separate validity/availability
masks. It contains 3,489 panel dates, 3,401 symbols, and 8,204,961 samples. All
2012-2025 samples have one source/train role; no fixed validation or test role is
created. Purged walk-forward builders own the later split and normalization logic.

Of the samples, 8,185,245 have a filled next-open entry and 19,716 unfilled
candidates remain in the ranking pool. Pack validation returned `status=ok` with
zero PIT price-coverage gaps. No model training, candidate selection, outer audit,
QDP activation, or execution promotion was performed.

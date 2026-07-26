# Quant Data Platform state

Updated: `2026-07-23`

- `active_as_of_date`: `2026-07-21`
- active dataset count: `14`
- quick check: `ok`
- full database audit: `ok`; the only finding is the medium historical 5-minute
  coverage warning for daily-only restored securities.
- PIT main-board inventory: 3,419 securities; 3,416 have daily rows; 378
  historical securities were restored, including 228 delisted securities.
- Historical ST state contains 193,539 rows across 342 securities.
- The two multi-ticker security identities are normalized to date-effective
  symbols in all daily research domains. The 5-minute dataset is byte-for-byte
  unchanged and is mapped only inside cross-frequency audit queries.
- `market_daily_raw`, `market_intraday_5m`, `adjust_factor`, calendar, status,
  universe, industry, and index tails reach 2026-07-21.
- `share_capital`, `valuation`, and corporate actions remain through 2026-07-16
  because strict PIT confirmation rejected unconfirmed provider changes.
- The daily PIT main-board scope is suitable for survivorship-bias-controlled
  research. Historical 5-minute coverage is intentionally not part of that
  contract.
- Legacy lake, ingest, memmap, event-pack builder, provider-evaluation, and v3
  bootstrap code were removed. Protected `data/qdp_v2` and `data/event_packs`
  remain unchanged; the current CLI exposes only status/list/describe/check,
  update, compact, and gc.

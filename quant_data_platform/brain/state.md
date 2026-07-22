# Quant Data Platform state

Updated: `2026-07-22`

- `active_as_of_date`: `2026-07-21`
- active dataset count: `14`
- quick check: `ok`
- `market_daily_raw`, `market_intraday_5m`, `adjust_factor`, calendar, status,
  universe, industry, and index tails reach 2026-07-21.
- `share_capital`, `valuation`, and corporate actions remain through 2026-07-16
  because strict PIT confirmation rejected unconfirmed provider changes.
- The current store is a user-selected current-survivor research store; it does
  not claim delisting-probability or survivorship-bias-free semantics.
- Legacy lake, ingest, memmap, event-pack builder, provider-evaluation, and v3
  bootstrap code were removed. Protected `data/qdp_v2` and `data/event_packs`
  remain unchanged; the current CLI exposes only status/list/describe/check,
  update, exclude, compact, and gc.

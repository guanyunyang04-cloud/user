# v2 Size/Turnover Source Audit Interpretation

- Date: 2026-06-02
- Run: `v2_industry_size_source_audit_20260602_235547`
- Snapshot: `baostock_v2_pit_20160101_20260601_industry_month_start_20260602`
- Probe: `sh.600000`, `2026-05-25` to `2026-06-01`

## Result

The current v2 snapshot still does not contain true market cap, float market cap, share base, or turnover fields.

Baostock live daily-history probe shows:

| Field | Available | Interpretation |
|---|---:|---|
| `turn` | yes | Daily turnover-rate candidate. |
| `turnover` | no | Invalid Baostock daily field name. |
| `turnover_rate` | no | Invalid Baostock daily field name. |
| `pctChg` | yes | Daily percent-change field candidate. |
| `peTTM` | yes | Valuation field candidate. |
| `pbMRQ` | yes | Valuation field candidate. |
| `psTTM` | yes | Valuation field candidate. |
| `pcfNcfTTM` | yes | Valuation field candidate. |

Representative evidence:

- `turn`: sample `["2026-05-25", "sh.600000", "9.0800", "0.278000"]`
- `peTTM`: sample `["2026-05-25", "sh.600000", "9.0800", "6.014658"]`
- `pbMRQ`: sample `["2026-05-25", "sh.600000", "9.0800", "0.401162"]`

## Interpretation

This opens a practical v2.1 extension path:

- Add a cached daily metrics table from Baostock history fields:
  `date, code, turn, pctChg, peTTM, pbMRQ, psTTM, pcfNcfTTM, source`.
- Use `turn` only after a full missingness and PIT timing audit.
- Use valuation fields only after auditing whether Baostock reports them with acceptable point-in-time semantics.

This does not solve true size neutralization:

- No true market cap or float market cap field is present in the current snapshot.
- No share-base field is present in the current snapshot.
- `amount`, `volume`, and `log_amount_mean_20d_z` remain liquidity/size proxies, not real market-cap controls.

Strategy candidate count remains `0`.

# PIT Signal Universe Internal Guide

- Status: historical and retired; not active
- Date: 2026-07-11
- Owner: `quant_data_platform`
- Audience: QDP and `daily_research` maintainers

> Superseded on 2026-07-17 by the user-selected mutable current-listed survivor
> scope. The candidate identifiers and instructions below are historical evidence;
> they are not current QDP objects and must not be added to `active.json`.

## Purpose

Use this scope when a historical experiment needs a date-local Shanghai/Shenzhen main-board universe without filtering history by the securities that survived to 2026. It does not replace the 3037-symbol core QDP scope. The candidate datasets have been materialized and audited, but the current `active.json` intentionally remains unchanged until the downstream data and pack gates pass.

## Candidate objects

- Eligibility: `pit_signal_universe__7c8d1f2986a646a5cb8a93a6`
- Daily audit: `pit_signal_universe_daily__7c8d1f2986a646a5cb8a93a6`
- Intended research-scope key after promotion: `research_scopes.pit_mainboard_non_st_v1`
- Source snapshot: `baostock_v2_pit_20160101_20260601_industry_metrics_month_start_20260603`

The row-level eligibility contract is:

```text
eligible_for_research
= listed on D
& Shanghai/Shenzhen main-board code
& common A share
& non-ST on D
& not delisted on D

eligible_for_signal
= eligible_for_research
& not suspended on D
& has a daily bar on D
```

A stock that delists later remains eligible on an earlier normal trading date. `out_date` and the source historical name are not emitted. This is deliberate: the source `name_on_date` contains retrospective/latest-name behavior, including 163,690 rows whose names contain `退`, so name-based historical filtering would leak future identity.

## Audit surface

`pit_signal_universe_daily` stores one row per date with:

- covered, research-eligible and signal-eligible counts;
- ST, suspension, missing-bar and delisted counts;
- `signal_membership_hash = md5(comma-joined eligible symbols sorted ascending)`.

The candidate build contains:

- 7,451,610 unique symbol-date rows;
- 2,526 trading dates from 2016-01-04 through 2026-06-01;
- 3,393 historical symbols;
- 3,392 symbols eligible at least once;
- 7,031,085 signal-eligible rows;
- 298,861 ST rows and 147,344 suspension rows;
- 375 historically eligible symbols absent from the latest eligible membership;
- zero source-tradeability semantic mismatches and zero post-delist rows.

## Rebuild and validation

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli rebuild pit-signal-universe --runtime fast --threads 4 --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli check --quick --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli status --verify-files --json
```

The build is content-addressed by source snapshot ID, source Parquet SHA-256, date bounds and both output contracts. A later explicitly gated activation can update the two dataset pointers and the research-scope record in one `active.json` write.

## Known limits

- The source ends on 2026-06-01, 18 open dates before the core QDP end date of 2026-06-26. It fully covers the planned 2018–2025 development and historical audit windows.
- Core OHLCV, intraday, limit and auxiliary QDP tables remain scoped to the current 3037-symbol survivor set. This eligibility table removes the universe-definition leak, but does not by itself restore missing multi-channel history for the additional securities.
- The source snapshot also contains daily bars for the same 3,393-symbol history. Importing/normalizing those prices is a separate data contract and must be validated before a corrected daily-only pack can claim full survivor-bias removal.
- Historical names and future `out_date` must never be added to model feature columns. They are source audit metadata only.

## Activation boundary

Do not add the `pit_signal_universe`, `pit_signal_universe_daily` or `research_scopes.pit_mainboard_non_st_v1` pointers to the current `active.json` until the additional historical securities' required price/feature channels and the downstream sequence pack pass their quality gates. Do not delete the candidate Parquet directories; they are preserved evidence and a rebuild input.

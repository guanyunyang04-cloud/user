# PIT L35V2 survivorship audit record

Status: completed on 2026-07-23.

The frozen legacy L35V2 six-fold result is severely affected by survivorship bias in the evaluation universe. Under the legacy-selected `Top1 / 1 slot / fixed D14 / double slippage` strategy, 2023-2025 liquidated ending equity falls from CNY 52,175,823.50 to CNY 13,816,328.42 on the complete PIT universe with legacy normalization, a 73.52% reduction. Annualized log growth falls from 1.37079 to 0.91020, with a paired 60-trading-day block-bootstrap difference of -0.46059 and 95% CI [-0.87084, -0.10624].

The 2020-2025 durability replay is more severe under the study's then-current zero-terminal-recovery convention: the complete PIT account ends at CNY 21.48 after `000018.SZ`, absent from the legacy pool, receives zero terminal recovery. Primary-period Rank IC does not flag the risk: it is 0.110743 in the legacy pool and 0.111212 in the complete PIT pool.

The retained report definition is [result.json](result.json). It contains 13 bounded datasets, 5 charts, 10 exact tables, metric definitions, uncertainty, sensitivity, limitations, and the next-step protocol.

The record retains the report data and scientific checks; process receipts and duplicate integrity logs were removed. No QDP dataset, research pack, or registered checkpoint was modified. Future retraining uses a 60-day purge.

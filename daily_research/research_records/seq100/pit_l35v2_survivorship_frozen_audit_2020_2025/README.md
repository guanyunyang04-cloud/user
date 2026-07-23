# PIT L35V2 survivorship audit record

Status: completed on 2026-07-23.

The frozen legacy L35V2 six-fold result is severely affected by survivorship bias in the evaluation universe. Under the legacy-selected `Top1 / 1 slot / fixed D14 / double slippage` strategy, 2023-2025 liquidated ending equity falls from CNY 52,175,823.50 to CNY 13,816,328.42 on the complete PIT universe with legacy normalization, a 73.52% reduction. Annualized log growth falls from 1.37079 to 0.91020, with a paired 60-trading-day block-bootstrap difference of -0.46059 and 95% CI [-0.87084, -0.10624].

The 2020-2025 durability replay is more severe: the complete PIT account ends at CNY 21.48 after `000018.SZ`, absent from the legacy pool, receives zero terminal recovery. Primary-period Rank IC does not flag the risk: it is 0.110743 in the legacy pool and 0.111212 in the complete PIT pool.

The interactive MCP report is defined by [artifact.json](artifact.json), SHA-256 `0c5dc29ea393ce212cab96329edfeffb0f092a6babd5694c1df93333fa8cbe10`. It contains 13 bounded datasets, 5 native charts, 10 exact tables, metric definitions, uncertainty, sensitivity, limitations, validation, and the next-step protocol.

Final validation is recorded in [final_validation/summary.json](final_validation/summary.json): frozen preflight, 30 inference manifests and 120 output hashes, account replay smoke, 17/17 independent calculations, 157 path-policy tests, QDP quick/full, Brain integrity, and `git diff --check` all passed. QDP full retained one accepted medium finding for unavailable historical 5-minute data on restored daily-only symbols. Brain integrity had zero errors and the two existing `outputs/` and `tmp/` root warnings.

No QDP dataset, research pack, or registered checkpoint was modified. The exact protected hashes are retained in [final_validation/protected_asset_preflight.json](final_validation/protected_asset_preflight.json). Future retraining uses a 60-day purge.

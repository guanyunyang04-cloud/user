# `a_stock_daily_selection` Historical Output Archive

- Status: `retired / output-only directory removed / not an active project or child brain`.
- Archived on: `2026-07-10`.
- Source tree contained no code, tests, README, manifest-backed data dependency, or registered brain—only eight tracked outputs totaling `322,246` bytes.
- Evidence boundary: the metrics below are historical output without a surviving reproducible pipeline in this workspace. They are not current research, execution, or promotion evidence.

## Preserved aggregate results

- Backtest rows: `795`, columns: `date, selected_count, gross_return, turnover, cost, net_return, selected_codes`.
- Date range starts at `2023-04-21`; the original daily row payload was retired with the directory.
- IC mean / rank IC mean: 1d `0.2441 / 0.2104`; 3d `0.2470 / 0.2043`; 5d `0.2668 / 0.2178`; 10d `0.2906 / 0.2302`; 20d `0.3017 / 0.2447`.
- Reported annual return `2.9841`, Sharpe `6.7738`, max drawdown `-0.0793`, average daily return `0.005586`, annual volatility `0.2078`, average turnover `0.6478`.
- Validation RMSE: 1d `0.02936`; 3d `0.05476`; 5d `0.07166`; 10d `0.10195`; 20d `0.14493`.

## Preserved recommendation identities

- `2026-05-07` (18): `600070.SH, 600016.SH, 000075.SZ, 600029.SH, 000032.SZ, 600072.SH, 000027.SZ, 000066.SZ, 600062.SH, 600045.SH, 600077.SH, 000073.SZ, 000022.SZ, 000028.SZ, 600085.SH, 000014.SZ, 000085.SZ, 000080.SZ`.
- `2026-05-08` (17): `600019.SH, 600070.SH, 600083.SH, 600086.SH, 600039.SH, 600021.SH, 600007.SH, 600087.SH, 000010.SZ, 600003.SH, 600063.SH, 600072.SH, 000022.SZ, 000025.SZ, 600057.SH, 600040.SH, 600041.SH`.

## Previously removed large artifacts

The retained 2026-05-11 manifest recorded four already-untracked payloads; none was present at retirement:

- `scored_predictions.pkl`: `75,839,422` bytes, SHA-256 `dd6a0b8e71762bac521ff7270c2186638522063aa4e0ee8bb57f30a86ed8e22b`.
- `feature_label_panel.pkl`: `68,262,917` bytes, SHA-256 `300376da911f35701d8d7a66b19c6033a8bcf4f57c310e49e4fac78e4bae7367`.
- `training_dataset.pkl`: `68,184,209` bytes, SHA-256 `d3fd10e7f74c2b256753065f0cdd249f778132f766666e7c1beb14ab51edee76`.
- `ranker.pkl`: `1,279,681` bytes, SHA-256 `e880549d8a50d00b0272e31cabd3d3c5a4577543dbee4b57687ed10bf72895f6`.

The historical conclusions remain discoverable here; recreating a runnable project would require a new explicit owner, source contract, code, tests, and data provenance.

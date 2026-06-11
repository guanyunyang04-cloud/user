# Canonical Data V1

`canonical_data_v1` is the workspace-level research base managed by `quant_data_platform`.

It includes raw daily bars, native/reused 5 minute bars, 5 minute derived daily features, adjust factors, valuation, industry, index constituents, trading calendar, universe snapshots, and security status.

It does not include financial quarterly reports, performance forecasts, or performance express reports in v1. Those domains need a separate point-in-time disclosure audit before becoming default research inputs.

Training profiles decide which fields are used:

- `short_horizon_core_v1`: market, intraday daily features, adjust factors, and tradeability filters.
- `style_structural_v1`: short core plus valuation, industry, and index membership.
- `medium_horizon_v1`: structural profile for longer lookback research.

Raw OHLCV is not overwritten by adjusted prices. Adjusted prices and returns stay as sidecars or derived features.

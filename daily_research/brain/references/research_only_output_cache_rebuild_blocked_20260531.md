# Research-Only Output / Cache Rebuild Blocked - 2026-05-31

## Scope
- Status: `research_only_rebuild_blocked / data_source_blocker / no model training launched`.
- User-selected route: rebuild only the research substrate needed to continue `alpha_multi_horizon_utility_policy_v1`, not full historical `output/cache` restoration.
- Active artifact impact: `daily_research/output/active_execution_strategy.json` unchanged.
- Boundary: this attempt does not restore live/default, daily execution payloads, active execution artifact, paper, broker, old study payloads, or old run-level file evidence.

## Actions Completed
- Workspace capsule routed the task to `daily_research`; active artifact guard was clean.
- CUDA preflight passed: `torch.cuda.is_available=true`, device `NVIDIA GeForce RTX 2060`.
- Six non-research `daily_research/output/test_*` brain fixture directories were checked for research payload markers and moved to `daily_research/cache/rebuild_quarantine/20260531_output_test_dirs/`.
- Provider health command completed with `status=degraded`; `market_daily`, `trading_calendar`, and `universe_snapshot` appeared available in aggregate, while several optional domains were missing or provider-specific errors.
- The full all-A refresh was launched for `2017-01-01 -> 2024-12-31` with required domains `market_daily,trading_calendar,universe_snapshot`.

## Blocker
- The refresh failed before any `policy_input_bundle` was registered.
- Failure point: `universe=all_a` resolution.
- Error: `ValueError: universe=all_a requires a non-empty universe_snapshot provider result`.
- Provider details:
  - `eastmoney_efinance`: `JSONDecodeError`.
  - `akshare_eastmoney`: `ProxyError` against Eastmoney API host.
  - `baostock`: `baostock_stock_basic_timeout: exceeded 60 seconds`.
- Log root: `daily_research/output/research_data_lake/rebuild_20260531/`.

## Result
- No new `policy_input_bundle__...` was created.
- No new rolling_liquid500 pool view or sector/board view was created.
- No full rolling_liquid500 memmap baseline was created.
- No multi-horizon baseline seed training was launched.
- No baseline continuation gate was evaluated.

## Interpretation
- This is a data-source availability blocker, not a model-quality result and not evidence that recent brain conclusions are wrong.
- The correct state remains: old conclusions are brain-confirmed/user-confirmed, while file-backed research rebuild is currently blocked by missing full all-A universe snapshot.
- Do not use partial universe, cap80, stale CSV, or manually reconstructed payloads to claim a full-pool research rebuild.

## Next Allowed Actions
- Retry provider health and all-A refresh only when universe snapshot providers are available.
- If a fresh, auditable CSV/universe source is supplied, import it through Bronze/Silver/lake first; do not train directly from scattered CSV.
- If a true backup exists, restoring `daily_research/output/` and `daily_research/cache/` remains the cleanest way to recover old file-backed evidence.
- Keep `daily_research/output/active_execution_strategy.json` untouched unless a future explicit promotion/recovery authority is given.

# Path Policy Test Runtime Optimization 20260621

## Verdict
- Status: `test_runtime_optimized / research_tooling / no_strategy_change`.
- Date: `2026-06-21`.
- Scope: `daily_research/path_policy/tests/test_rl_protocol.py` protocol test runtime optimization.
- Active artifact impact: none. No `daily_research/output/active_execution_strategy.json`, live/default, paper/live/broker, score-backtest bridge, candidate matrix, execution-candidate state, QDP canonical data, or QDP registry active pointer was changed.

## Files Changed

```text
daily_research/path_policy/tests/conftest.py
daily_research/path_policy/tests/test_rl_protocol.py
```

## What Changed

The old whole-file slow behavior in `test_rl_protocol.py` was split into fast protocol contracts and retained slow end-to-end smoke tests.

Implemented changes:

```text
test_rl_protocol.py module marker changed from research+slow to research
only true end-to-end protocol fixtures retain slow/integration/training markers
forecast e2e fixtures use smaller synthetic date ranges, horizon=1, lookback=2, epochs=1, seed=7, and smaller role caps
manifest reuse tests share a tiny legacy memmap manifest fixture instead of rebuilding repeatedly
matrix orchestration and v5 long-context incomplete tests were converted to targeted logic/diagnostic contracts
```

Retained real slow smoke coverage:

```text
forecast eager contract
forecast memmap contract
forecast manifest reuse fast path
forecast manifest CLI fast path
RL walkforward
v4 validation repair
v5 DT validation
```

## Verification

Before optimization:

```text
pytest daily_research/path_policy/tests/test_rl_protocol.py -q
59 passed in 440.51s
```

After optimization:

```text
pytest daily_research/path_policy/tests/test_rl_protocol.py -q -m "not slow" --durations=8
52 passed, 7 deselected in 13.65s

pytest daily_research/path_policy/tests/test_rl_protocol.py -q -m slow --durations=15
7 passed, 52 deselected in 154.85s

pytest daily_research/path_policy/tests/test_rl_protocol.py -q --durations=15
59 passed in 162.22s

git diff --check
passed
```

## Interpretation

This supersedes the stale note in `path_policy_alpha_v2_date_slate_incremental_implementation_20260621.md` that `test_rl_protocol.py` needs about 7.5 minutes. Current observed full-file runtime is about 2m42s; normal non-slow protocol checks are about 14s.

This is research tooling evidence only. It does not change model behavior, model quality evidence, active artifacts, registry pointers, promotion gates, or execution behavior.

## Next Allowed Actions

```text
Use pytest daily_research/path_policy/tests/test_rl_protocol.py -q -m "not slow" for fast protocol checks.
Use pytest daily_research/path_policy/tests/test_rl_protocol.py -q -m slow for retained end-to-end forecast/RL protocol smoke.
Use full test_rl_protocol.py when validating protocol/test refactors that may affect both fast contracts and slow smoke.
```

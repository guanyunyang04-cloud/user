# Data Lake Universal Repair Contract - 2026-05-17

## Summary
- Status: `current infrastructure mainline / data_lake_universal_repair / research / shadow-only / no promotion`.
- The current work pointer is temporarily switched from Path20 Stage 1 training to universal data-lake repair because the base policy input bundle must be trustworthy before forecast evidence is interpretable.
- Boundary: this contract repairs data correctness and auditability only. It does not change live/default execution and does not modify `daily_research/output/active_execution_strategy.json`.

## Facts
- `policy_input_bundle__0f116a9b78c92ff045a6853d` covers `2010-01-04 -> 2026-05-13` and `learned_all_a` uncapped with about `3070` symbols, but its `000300.SH` benchmark close coverage is missing before `2018-05-14`.
- `policy_input_bundle__4db1a32ab6e7d77ac7b8671c` is usable from `2018-05-14 -> 2026-05-13`, but it is capped at `1200` symbols and is not full-universe evidence.
- Prior policy-input bundles stored only benchmark `close`; lake loading silently fell back from missing benchmark `open` to `close`, which is not strict enough for Path20 `next_open` labels.
- The data-lake interface now supports persisted benchmark `open` in `silver_benchmark.parquet`.
- `load_policy_inputs_from_lake(..., require_benchmark_open=True)` blocks old close-as-open fallback with `missing_benchmark_open`.
- Path20 `next_open` protocol preparation now requires true lake benchmark open.
- New audit entrypoint: `python -m daily_research.data_lake.policy_input_audit`.
- Cap80 repaired smoke bundle completed:
  - dataset id: `policy_input_bundle__c4886777fe70bfe6616e1259`
  - window `2010-01-04 -> 2026-05-13`
  - universe size `80`
  - strict audit verdict with benchmark open required: `usable`
- Full-universe repaired bundle completed:
  - dataset id: `policy_input_bundle__7c8f58d851bce8179e1e9e2d`
  - window `2010-01-04 -> 2026-05-13`
  - universe size `3070`
  - feature panels `109`
  - benchmark close rows `3969`, benchmark open rows `3969`
  - strict audit verdict for `2019-01-01 -> 2024-12-31`: `usable`
- Research lake default policy input id is now `policy_input_bundle__7c8f58d851bce8179e1e9e2d`.

## Blockers
- Do not run Path20 Stage 1 formal training on `policy_input_bundle__0f116a9b78c92ff045a6853d` for `2019-2024`; use the repaired full bundle instead.
- Do not treat `policy_input_bundle__4db1a32ab6e7d77ac7b8671c` as full-universe evidence; it is capped and starts at `2018-05-14`.
- Do not run full-universe Path20 forecast training with the eager `[N,252,F]` dataset until memory-safe sequence loading exists.

## Next Allowed Actions
- Use `policy_input_bundle__7c8f58d851bce8179e1e9e2d` for future repaired full-universe policy-input audits and capped Path20 Stage 1 pilots.
- Do not run full-universe Path20 Stage 1 training until memory-safe sequence loading exists.
- If running capped Stage 1 pilots, keep explicit `--lake-dataset-id policy_input_bundle__7c8f58d851bce8179e1e9e2d`.
- Keep `active_execution_strategy.json` unchanged throughout this repair line.

## Active Artifact Impact
- unchanged

## Dataset IDs
- `policy_input_bundle__0f116a9b78c92ff045a6853d`
- `policy_input_bundle__4db1a32ab6e7d77ac7b8671c`
- `policy_input_bundle__c4886777fe70bfe6616e1259`
- `policy_input_bundle__7c8f58d851bce8179e1e9e2d`

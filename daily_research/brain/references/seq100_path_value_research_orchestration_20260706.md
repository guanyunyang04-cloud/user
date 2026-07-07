# Seq100 Path-Value Research Orchestration

日期：`2026-07-06`

## Verdict

`daily_research` owns the current seq100 path-value research line. QDP owns the active data base that supplies source facts, but model-ready sequence packs, labels, normalization, model outputs, path-value semantics and evaluation conclusions are `daily_research` research artifacts.

## Current Research Facts

- Current line: `seq100_path_value_research`.
- Input principle: past 100 trading days of daily raw/state, intraday summary and limit-structure sequences.
- Output principle: predict future path; derive path summary and path trade value from predicted path.
- Compared model families: base GRU, path-only GRU, residual-score, OHLCVA, today-close anchor.
- Evidence status: today-close anchor improved test rank IC versus next-open path-only, but topK concentration is not yet decisive.
- Promotion status: research evidence only; no active execution strategy or live/default pointer change.

## Ownership Decision

- QDP active data base: `quant_data_platform/data/qdp_v2/active/active.json`, dataset manifests, parquet shards, provider ingest and quality proofs.
- Research artifacts: sequence packs, memmaps, normalization, sample index, labels, model outputs, prediction CSVs, summaries and evaluations.
- Artifact root: `daily_research/data/research_store/<artifact_id>/`.
- Legacy QDP research roots were physically migrated or deleted on 2026-07-07; they are not valid active entrypoints.

## Resource Decision

H: space pressure is mainly from repeated research packs and path-policy study outputs. Active QDP data base size is not the primary pressure source. Cleanup should start with GC dry-run over smoke packs, failed partial packs, duplicate research pack inputs and large stale prediction outputs, not with QDP active datasets.

## Brain System Decision

Route/capsule should expose primary owner, supporting brains, object routes and writeback targets. Mixed QDP + model/research tasks should select primary `daily_research` with supporting read-only `quant_data_platform`, unless the task actually changes active QDP datasets or quality facts.

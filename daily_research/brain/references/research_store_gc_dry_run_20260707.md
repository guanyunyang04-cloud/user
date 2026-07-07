# Research Store GC Dry Run 2026-07-07

## Context
The H: space pressure is mainly from downstream research artifacts, not the QDP active data base. QDP owns active manifests, parquet datasets, provider ingest and quality proofs. `daily_research` owns sequence packs, model-ready memmaps, labels, normalization, training outputs, studies and reports.

New model-ready artifacts now default to:

```text
daily_research/data/research_store/sequence_pack
```

Historical sequence packs under this compatibility root remain readable but are not QDP active data base objects:

```text
quant_data_platform/data/qdp_v2/research/sequence_pack
```

## Implemented Controls
- Added `daily_research.path_policy.research_store_gc`.
- Default mode is dry-run.
- Directory deletion requires `--delete --confirm-delete DELETE_RESEARCH_ARTIFACTS`.
- Active QDP datasets are not delete candidates.
- Safe directory deletion is limited to unreferenced smoke, partial or interrupted research artifacts under configured research roots.
- Full packs and reanchor/view packs are kept for review.
- Large prediction CSV/parquet outputs are reported as trim candidates, not deleted by the directory GC path.
- `qdp_v2_sequence_path_pack.DEFAULT_OUTPUT_ROOT` now points to `daily_research/data/research_store/sequence_pack`.

## Dry-Run Result
Command:

```powershell
conda run -n yolos python -m daily_research.path_policy.research_store_gc scan --write-report --max-items 30 --json
```

Report:

```text
daily_research/output/path_policy/research_gc/research_gc_dry_run_20260707_014121.md
daily_research/output/path_policy/research_gc/research_gc_dry_run_20260707_014121.json
```

Scanned artifacts:

```text
artifact_count: 231
total_size: 149.3648 GB
safe_delete_candidate_count: 74
safe_delete_candidate_size: 14.4109 GB
prediction_trim_candidate_count: 46
prediction_trim_candidate_size: 63.5546 GB
```

Largest classification buckets:

```text
study_with_large_prediction_outputs: 46 / 75.5682 GB
full_sequence_pack: 3 / 37.8320 GB
view_or_reanchor_sequence_pack: 1 / 14.3887 GB
smoke_sequence_pack: 5 / 10.0714 GB
smoke_or_throughput_study: 52 / 6.4409 GB
partial_or_intermediate_study: 88 / 3.8934 GB
study_with_summary: 36 / 1.1703 GB
```

## Interpretation
The immediate safe reclaim candidate is modest but real: about 14.41GB from unreferenced smoke/partial/interrupted directories. The larger opportunity is prediction-output trimming, about 63.55GB, but that should be a separate guarded command that preserves summaries, metrics, best checkpoints and report files before removing large forecast CSV/parquet files.

No data was deleted during this run.

## Next Method
Review the generated dry-run report before deletion. If accepted, run only the guarded directory-delete path for safe candidates first. Add a separate `trim-predictions` command before removing large prediction outputs.

## Executed Cleanup
User approved the guarded cleanup path on 2026-07-07.

Executed command:

```powershell
conda run -n yolos python -m daily_research.path_policy.research_store_gc scan --write-report --max-items 1000 --delete --confirm-delete DELETE_RESEARCH_ARTIFACTS --json
```

Result:

```text
deleted_count: 74
deleted_size: 14.4109 GB
```

Deleted artifact classes:

```text
unreferenced smoke sequence packs
unreferenced smoke/throughput studies
unreferenced partial/intermediate studies
```

The deleted sequence packs were only smoke packs:

```text
smoke_seq100_ohlcva_path60_trainvaltest_codex
smoke_seq100_ohlcva_path60_codex
smoke_seq100_path60_trainable
smoke_seq100_path60
smoke_seq100_path20
```

Post-delete dry-run:

```text
daily_research/output/path_policy/research_gc/research_gc_dry_run_20260707_075934.md
artifact_count: 157
total_size: 134.9540 GB
safe_delete_candidate_count: 0
prediction_trim_candidate_count: 46
prediction_trim_candidate_size: 63.5546 GB
```

## Zero-Copy Legacy Views
Created daily_research-owned view manifests for kept legacy sequence packs. This did not copy large arrays and did not move physical data.

Command:

```powershell
conda run -n yolos python -m daily_research.path_policy.research_store_gc register-legacy-views --write-report --json
```

View registration report:

```text
daily_research/output/path_policy/research_gc/legacy_sequence_pack_views_20260707_080158.json
```

Views:

```text
daily_research/data/research_store/sequence_pack/qdp_v2_seq100_ohlcva_path60_full/manifest.json
daily_research/data/research_store/sequence_pack/qdp_v2_seq100_path20_full/manifest.json
daily_research/data/research_store/sequence_pack/qdp_v2_seq100_path60_full/manifest.json
daily_research/data/research_store/sequence_pack/qdp_v2_seq100_path60_todayclose_full/manifest.json
```

Validation:

```text
qdp_v2_seq100_ohlcva_path60_full: ok / 6,398,421 samples
qdp_v2_seq100_path20_full: ok / 6,620,640 samples
qdp_v2_seq100_path60_full: ok / 6,398,421 samples
qdp_v2_seq100_path60_todayclose_full: ok / 6,398,421 samples
```

Final dry-run after view registration:

```text
daily_research/output/path_policy/research_gc/research_gc_dry_run_20260707_080253.md
artifact_count: 161
total_size: 134.9545 GB
safe_delete_candidate_count: 0
prediction_trim_candidate_count: 46
prediction_trim_candidate_size: 63.5546 GB
```

Next cleanup method: add a separate prediction-output trim guard that preserves summary, metrics, best checkpoints and reports before deleting large forecast CSV/parquet files.

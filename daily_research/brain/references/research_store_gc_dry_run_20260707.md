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

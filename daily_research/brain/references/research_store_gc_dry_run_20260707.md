# Research Store GC Dry Run 2026-07-07

## Context
The H: space pressure is mainly from downstream research artifacts, not the QDP active data base. QDP owns active manifests, parquet datasets, provider ingest and quality proofs. `daily_research` owns sequence packs, model-ready memmaps, labels, normalization, training outputs, studies and reports.

New model-ready artifacts now default to:

```text
daily_research/data/research_store/sequence_pack
```

Historical research artifacts under `quant_data_platform/data/qdp_v2/research/` were wrong-owner model-ready artifacts. They have been physically migrated to `daily_research/data/research_store/` or deleted when they were smoke/partial artifacts.

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

## Physical Legacy Migration
The temporary view step was superseded by physical migration because training artifacts should live where their owner lives.

Command:

```powershell
conda run -n yolos python -m daily_research.path_policy.research_store_gc migrate-legacy-packs --write-report --json
```

Sequence pack migration report:

```text
daily_research/output/path_policy/research_gc/legacy_sequence_pack_migration_20260707_081550.json
```

Migrated sequence packs:

```text
qdp_v2_seq100_ohlcva_path60_full
qdp_v2_seq100_path20_full
qdp_v2_seq100_path60_full
qdp_v2_seq100_path60_todayclose_full
```

The old alpha_v2 sharded memmap and training pack were also physically moved:

```text
daily_research/data/research_store/sharded_memmap/qdp_v2_alpha_v2_full_contract_2012_2025_20260702_01
daily_research/data/research_store/training_pack/qdp_v2_alpha_v2_full_contract_2012_2025_20260702_01_training_pack
```

Deleted smoke artifacts from the old QDP research root:

```text
qdp_v2_alpha_v2_smoke_pack_contract_20260702_01
qdp_v2_alpha_v2_smoke_shard_20260702_01
qdp_v2_alpha_v2_smoke_shard_contract_20260702_01
qdp_v2_alpha_v2_smoke_pack_contract_20260702_01_training_pack
```

Validation after physical migration:

```text
qdp_v2_seq100_ohlcva_path60_full: ok / 6,398,421 samples
qdp_v2_seq100_path20_full: ok / 6,620,640 samples
qdp_v2_seq100_path60_full: ok / 6,398,421 samples
qdp_v2_seq100_path60_todayclose_full: ok / 6,398,421 samples
actual pathlike strings under daily_research/data/research_store: 5,220 checked / 0 missing
old QDP research root: missing
old QDP research path strings under research_store: 0
```

Final dry-run after physical migration:

```text
daily_research/output/path_policy/research_gc/research_gc_dry_run_20260707_082023.md
artifact_count: 157
total_size: 134.9540 GB
safe_delete_candidate_count: 0
prediction_trim_candidate_count: 46
prediction_trim_candidate_size: 63.5546 GB
```

## Prediction Output Trim
The separate prediction-output trim guard was implemented and executed on 2026-07-07.

Dry-run command:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.research_store_gc trim-predictions --write-report --json
```

Dry-run result:

```text
daily_research/output/path_policy/research_gc/prediction_trim_dry_run_20260707_085158.md
candidate_study_count: 46
candidate_file_count: 92
candidate_size: 63.5546 GB
```

Executed command:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.research_store_gc trim-predictions --delete --confirm-trim TRIM_PREDICTION_OUTPUTS --write-report --json
```

Executed result:

```text
daily_research/output/path_policy/research_gc/prediction_trim_execute_20260707_085215.md
deleted_file_count: 92
deleted_size: 63.5546 GB
skipped_file_count: 0
```

Scope:

```text
deleted: large prediction CSV/parquet/feather outputs under daily_research/output/path_policy/studies
retained: study summaries, split/topK/daily IC metrics, training history, reports, checkpoints
per-study evidence: prediction_trim_manifest.json
```

Post-trim dry-run:

```text
daily_research/output/path_policy/research_gc/research_gc_dry_run_20260707_085231.md
artifact_count: 157
total_size: 71.3996 GB
safe_delete_candidate_count: 0
prediction_trim_candidate_count: 0
prediction_trim_candidate_size: 0 GB
```

Prevention changes:

```text
qdp_v2_sequence_path_training default prediction mode: compact
full path-level prediction CSVs now require: --prediction-mode full --allow-large-predictions
qdp_v2_sequence_flat_lgbm no longer writes predictions by default; requires --write-predictions
```

## Unified Research Store Views
The old self-contained sequence packs were replaced by shared store components and lightweight view manifests on 2026-07-07.

Commands:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.research_store_view build-from-legacy-packs --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.research_store_view verify --max-checks 256 --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.research_store_view delete-replaced-legacy-packs --confirm-delete DELETE_REPLACED_SEQUENCE_PACKS --json
```

New entrypoints:

```text
daily_research/data/research_store/views/seq100_path60_nextopen_ohlc.json
daily_research/data/research_store/views/seq100_path60_nextopen_ohlcva.json
daily_research/data/research_store/views/seq100_path60_todayclose_ohlcva.json
daily_research/data/research_store/views/seq100_path20_nextopen_ohlc_from_path60.json
```

Storage layout:

```text
panel_store: shared historical input panels and masks
label_store: path60 next-open OHLC, path60 next-open OHLCVA, path60 today-close OHLCVA, path20 next-open OHLC
sample_index: path60 sample index and path20 sample index remapped onto the shared panel store
views: lightweight trainable manifests
```

Important semantic finding:

```text
path20 cannot be fully replaced by slicing path60 labels.
Reason: early path20-valid samples, starting around date_idx=99 / 2012-04-23, have finite 20-day labels while the path60 label array is still NaN for those rows.
Resolution: path20 view shares input panels but keeps an independent path20 label store and uses label_symbol_idx for label lookup.
```

Validation:

```text
research_store_view verify: ok
path20 equivalence checks: 256
path20 sample_count: 6,620,640
path20 label source: moved_label_store
SequencePathPackDataset smoke:
  path20 view -> x [2,100,84], y_path [2,20,4], y_summary [2,12]
  todayclose OHLCVA view -> x [2,100,84], y_ohlcva [2,60,6]
```

Deleted legacy full pack directories:

```text
qdp_v2_seq100_path60_full
qdp_v2_seq100_ohlcva_path60_full
qdp_v2_seq100_path60_todayclose_full
qdp_v2_seq100_path20_full
```

Post-unification scan:

```text
daily_research/output/path_policy/research_gc/research_gc_dry_run_20260707_091946.md
artifact_count: 162
total_size: 94.2894 GB
research_store_shared_component: 75.1106 GB
safe_delete_candidate_count: 0
prediction_trim_candidate_count: 0
H: free space after cleanup/unification: about 567.21 GB
```

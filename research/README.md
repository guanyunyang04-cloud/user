# Research records

`studies/` contains durable study inputs and `records/` contains retained
results and conclusions. These are evidence and history, not importable
runtime code. Active model execution is implemented in
`src/quantlab/research` and writes to `runs/`.

Study specifications use workspace-relative POSIX paths. References that were
resolvable after the `daily_research`/`quant_data_platform` migration now point
to `research/`, `data/qdp/`, or `runs/`. A `legacy://...` value is an explicit
historical provenance reference to an artifact that is no longer present; it
must not be opened as a current input or silently replaced with a guessed file.
The complete field-level migration record is
`research/path_migration_manifest.json`.

Before adding or reactivating a study, check its path contract with:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe tools/research_path_migration.py `
  --workspace-root H:/quant_project --check
```

Older records may retain the original `daily_research` or
`quant_data_platform` path strings as historical provenance. They are not
runtime imports. The large legacy output trees were purged from the external
archive on 2026-08-21; Git retains tracked source history and the archive
retains only small audit, migration, and legacy-provenance files.

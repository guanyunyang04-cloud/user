# Research records

`studies/` contains durable study inputs and `records/` contains retained
results and conclusions. These are evidence and history, not importable
runtime code. Active model execution is implemented in
`src/quantlab/research` and writes to `runs/`.

Older records may retain the original `daily_research` or
`quant_data_platform` path strings as historical provenance. They are not
runtime imports. The large legacy output trees were purged from the external
archive on 2026-08-21; Git retains tracked source history and the archive
retains only small audit, migration, and legacy-provenance files.

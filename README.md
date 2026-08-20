# QuantLab

QuantLab is a personal A-share research workspace. The active code is one
installable package under `src/quantlab`; historical experiments are outside
the execution path and current data products are under `data/`.

## Layout

- `src/quantlab/data`: point-in-time market-data acquisition, repair, catalog,
  and quality checks.
- `src/quantlab/research`: daily cross-sectional models, sequence models,
  portfolio replay, and research-data contracts.
- `data/qdp`: the mutable QDP data lake and its active manifests.
- `data/research`: the compact research dataset used by the current studies.
- `research`: durable study configurations, records, and historical model
  notes.
- `runs`: reproducible model and evaluation outputs.
- `tools`: one-off migration and maintenance utilities.

The previous `daily_research/` and `quant_data_platform/` trees are no longer
active. Git retains their tracked history; the bulky ignored experiment
outputs that had been staged in `H:\quant_project_archive\20260820` were
purged on 2026-08-21 to make room for market-data downloads. Small audit,
migration, and legacy-provenance files remain in that directory.

## Environment

Use the project environment from PowerShell:

```powershell
$env:PYTHONPATH='H:\quant_project\src'
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quantlab data status --verify-files
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quantlab data check --quick --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quantlab research verify
```

The current research dataset covers 4,191,476 stock-days through 2025-12-31,
with 183 validated input fields and no permitted 2026 outcome reads. The
current model contract and empirical conclusions are summarized in
[`CURRENT.md`](CURRENT.md); detailed evidence remains in `research/records/`
and the manifests beside each data product.

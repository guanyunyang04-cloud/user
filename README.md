# QuantLab

QuantLab is a personal A-share research workspace. The active code is one
installable package under `src/quantlab`; historical experiments are outside
the execution path and current data products are under `data/`.

## Layout

- `src/quantlab/data`: point-in-time market-data acquisition, repair, catalog,
  and quality checks.
- `src/quantlab/data/provider_symbols.py` and `src/quantlab/data/identifiers.py`:
  shared provider-code and security-identity mappings.
- `src/quantlab/data/qdp_v2/normalization.py` and
  `src/quantlab/data/qdp_v2/pit_normalization.py`: deterministic provider and
  PIT transformations kept independent from network/database orchestration.
- `src/quantlab/research`: daily cross-sectional models, sequence models,
  portfolio replay, and research-data contracts.
- `data/qdp`: the mutable QDP data lake and its active manifests.
- `data/research`: the compact research dataset used by the current studies.
- `research`: durable study configurations, records, and historical model
  notes.
- `runs`: reproducible model and evaluation outputs.
- `tools`: one-off migration and maintenance utilities.

The default incremental data path uses BaoStock for structured daily facts,
MootDX for recent intraday/corporate-action detection, and CNInfo through
AkShare for disclosure confirmation. Tushare-compatible code is retained only
as an explicitly selected legacy repair path. iQuant is treated as a runtime
and execution source, not as the sole historical research store. A read-only
adapter now decodes its local daily/one-minute K-line files and compares them
with QDP without an RPC or trading connection. The 2026-08-21 parity snapshot
found excellent recent agreement but only 60 one-minute files matching the
3,416 symbols with QDP daily bars; the download was still active. The cache is
therefore a recent/live supplement and independent validator, not a historical
replacement. Compact evidence is retained in
`research/records/iquant_cache_parity_20260821/result.json`.

PIT restore provenance in the active QDP manifest is stored relative to the
workspace (`path_base=workspace_root`), so moving the project does not leave
stale machine-specific paths. The semantic audit distinguishes explicit
quality checks from older provenance-only workflow records.

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

Local iQuant K-line files can be sampled without a trading connection. The
adapter validates the binary layout and records all inferred units before
comparing the sample with the active QDP store:

```powershell
$env:PYTHONPATH='H:\quant_project\src'
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quantlab data iquant-parity `
  --data-dir 'H:\国信iQuant策略交易平台\datadir'
```

The current research dataset covers 4,191,476 stock-days through the cutoff
declared in `data/research/daily/manifest.json`, with 183 validated input
fields. Outcome rows after that cutoff are rejected. The current model
contract and empirical conclusions are summarized in
[`CURRENT.md`](CURRENT.md); detailed evidence remains in `research/records/`
and the manifests beside each data product.

# QuantLab

QuantLab is a personal A-share research workspace. The active code is one
installable package under `src/quantlab`; historical experiments are outside
the execution path and current data products are under `data/`.

## Layout

- `src/quantlab/core`: shared file hashing, atomic file installation, paths,
  and research artifact I/O.
- `src/quantlab/data/domains/contracts`: domain requests, schemas,
  normalization dispatch, and coverage checks.
- `src/quantlab/data/providers`: provider registry plus focused BaoStock,
  MootDX, CNInfo, and web adapters. Transport is separate from frame shaping.
- `src/quantlab/data/qdp_v2`: the point-in-time store. Multi-stage workflows
  use focused `config`, `context`, `download`, `prepare`, `install`, `audit`,
  and `workflow` modules rather than monolithic scripts.
- `src/quantlab/data/qdp_v2/normalization.py` and
  `src/quantlab/data/qdp_v2/pit_normalization.py`: deterministic provider and
  PIT transformations independent from network/database orchestration.
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

The former large Python modules were replaced by packages at the same import
paths. Their `__init__.py` files preserve the established callable API while
the implementation is divided by responsibility. Repeated SHA-256 and atomic
installation logic is centralized. Production orchestration functions are
bounded and focused; the sole function over 100 lines is a named SQL query
builder kept intact so its relational logic can be reviewed as one statement.

Cross-session project state lives only in [`CURRENT.md`](CURRENT.md). The old
project `brain/` directory and user-level `workspace-brain` skill are removed;
normal Codex context plus this concise handoff file are sufficient.

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

Downloaded minute archives are managed as immutable QDP sources under
`data/qdp/source_archives/minute/`; they are never unpacked into a second CSV
tree or written into a broker cache. `quantlab.data.minute_archive` can create
short-lived research extracts, but its formal path streams a complete year
directly from ZIP into the canonical one-minute store.

```powershell
$env:PYTHONPATH='H:\quant_project\src'
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quantlab.data.minute_archive import-year `
  --archive 'H:\quant_project\data\qdp\source_archives\minute\1分钟(2000-2025).zip' `
  --year 2025 `
  --workspace-root 'H:\quant_project'
```

The completed 2025 pilot stores 09:30 rows in `market_opening_auction` and the
240 continuous bars in `market_intraday_1m`. Repeated share fields are retained
once per stock-day as audit evidence, not duplicated into every minute. The
quality directory records member CRCs, session completeness, daily parity and
the sparse stock-day exclusion list. See [`CURRENT.md`](CURRENT.md) for the
measured quality and research findings.

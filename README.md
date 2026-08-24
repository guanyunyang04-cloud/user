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

The `qdp_v2` names describe the current legacy layout, not a second supported
data generation. Its physical data is healthy, but the namespace, active
generation pointer, and hashed directories are scheduled for an in-place
flattening migration. The target retains one compact manifest per domain and
atomic installation; it does not retain parallel `v1/v2/v3` stores.

The default incremental data path uses BaoStock for structured daily facts,
MootDX for recent intraday/corporate-action detection, and CNInfo through
AkShare for disclosure confirmation. A Tushare-compatible profile can be
explicitly selected as a secondary repair and cross-validation source, but it
is never the sole authority for installing QDP facts. The current weekly token
has been verified against `stk_mins` and supplies A-share stock history at
1/5/15/30/60-minute frequencies from at least 2009, in addition to candidate
daily, factor, valuation, PIT-financial, margin, report and corporate-action
rows. ETF, index, futures and options minute interfaces remain separately
unauthorized. Minute rows are repair candidates rather than blanket
replacements because sampled historical anomalies often match the existing ZIP
source. Three audited price-only repair batches have installed 5,048 targeted
minute rows while retaining QDP volume and amount; the continuous/auction row
counts and primary keys are unchanged. Raw responses, hashes, decisions,
backups, and annual re-audits live under
`data/qdp/source_archives/tushare_compatible/minute_repair/`. The provider also
enforces a separate 20,000-call daily limit for `stk_mins`, and HTTP 429 is a
terminal, resumable condition rather than an item-by-item retry. iQuant is
treated as a runtime and execution source, not as the sole
historical research store. A read-only adapter now
decodes its local daily/one-minute K-line files and compares them with QDP
without an RPC or trading connection. The 2026-08-21 parity snapshot found
excellent recent agreement but only 60 one-minute files matching the 3,416
symbols with QDP daily bars; the download was still active. The cache is
therefore a recent/live supplement and independent validator, not a historical
replacement. Compact evidence is retained in
`research/records/iquant_cache_parity_20260821/result.json` and
`research/records/tushare_compatible_provider_probe_20260823/result.json`.

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
short-lived research extracts, while its formal path streams selected years
directly from ZIP into transactionally appended year/month Parquet partitions.
For large imports, streaming and auditing are deliberately split across two
processes so Windows releases every Arrow file handle before validation.

```powershell
$env:PYTHONPATH='H:\quant_project\src'
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quantlab.data.minute_archive stage-years `
  --archive 'H:\quant_project\data\qdp\source_archives\minute\1分钟(2000-2025).zip' `
  --start-year 2010 `
  --end-year 2025 `
  --workspace-root 'H:\quant_project'

# Pass the returned staging path to a fresh process.
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quantlab.data.minute_archive resume-staged `
  --staging '<returned staging path>' `
  --workspace-root 'H:\quant_project'
```

The canonical store now covers 2010-01-04 through 2026-08-21: 3,301,495,126
continuous bars and 13,756,395 separate 09:30 rows in 200 monthly shards per
domain. Repeated share fields are retained once per stock-day as audit evidence,
not duplicated into every minute. Annual quality records cover member parsing,
session completeness, primary keys, deterministic repairs, daily parity, and
sparse stock-day exclusion lists. The importer sizes its buffers from currently
available physical memory, reserves at least 25% of RAM (and at least 3 GiB) for
the operating system, and flushes early if free memory falls below that reserve.
See [`CURRENT.md`](CURRENT.md) for the measured quality and known source limits.

Targeted repairs operate only on stock-days already flagged by the annual
minute audit. The workflow reuses valid raw captures, caps each range at 33
actual trading days (7,953 possible rows), requires an exact 241-row session,
and overlays accepted price fields without replacing volume or amount:

```powershell
$env:PYTHONPATH='H:\quant_project\src'
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quantlab data minute-repair `
  --selection-mode open `
  --minimum-relative-error 0.005 `
  --maximum-relative-error 0.01 `
  --max-trading-days 33 `
  --workspace-root 'H:\quant_project'
```

Omit `--apply` for candidate evaluation. Add it only after reviewing the
generated `result.json`; valid raw captures make the command resumable.

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
source. The first three audited price-only repair batches installed 5,048
targeted minute rows. Six later explicit fallback runs evaluated the local
archive's unresolved queue and installed another 8,238 rows, bringing the
cumulative Tushare-compatible total to 13,286. QDP volume and amount were
retained; the continuous/auction row counts and primary keys are unchanged.
Compact targets, hashes, decisions and annual re-audit results live under
`data/qdp/source_archives/tushare_compatible/minute_repair/`; bulky runtime
captures and full pre-install copies may be discarded after verification. The provider also
enforces a separate 20,000-call daily limit for `stk_mins`, and HTTP 429 is a
terminal, resumable condition rather than an item-by-item retry. iQuant is
retained only as a read-only recent-data validator, not as the planned
execution platform. A read-only adapter now
decodes its local daily/one-minute K-line files and compares them with QDP
without an RPC or trading connection. The 2026-08-21 parity snapshot found
excellent recent agreement but only 60 one-minute files matching the 3,416
symbols with QDP daily bars; the download was still active. The cache is
therefore a recent/live supplement and independent validator, not a historical
replacement. Compact evidence is retained in
`research/records/iquant_cache_parity_20260821/result.json` and
`research/records/tushare_compatible_provider_probe_20260823/result.json`.
Future execution integration is deferred until a broker QMT terminal with the
required API entitlement is installed; Guojin QMT is the provisional target.

The purchased local archive at `H:\BaiduNetdiskDownload\量化数据\stock_1min`
is used only as a historical minute-repair source. Its inventory contains
5,826 canonical per-symbol Parquet files; 43 parenthesized duplicate files are
explicitly excluded. The adapter reads each symbol once, normalizes the source
schema, extracts only audited stock-days, and reuses the existing 241-row,
daily-parity, OHLC, flow-preservation, CAS-install and annual re-audit
gates. Two sequential local-source batches installed 6,172 open-row repairs
and 108,161 high/low row-level repairs. Two later relative-error/own-flow runs
installed another 17,974 rows, bringing the purchased-source total to 132,307.
The final audit has 55,104 field-level price-excluded stock-days, 0.4006% of
the 13,756,395 audited stock-days, plus 125 independent session exclusions.
The relative-error tiers are 35,548 in `(0.5%, 1%]`, 15,429 in `(1%, 2%]`,
3,693 in `(2%, 5%]`, 402 in `(5%, 10%]`, and 32 above 10%. Volume and amount
were not replaced. The high/low timing gate
passed: the local one-minute source reproduced the trusted daily extreme on
261/263 samples. Independently reliable BaoStock sessions placed it in the
same five-minute bucket in 52/53 cases. Evidence is under
`data/qdp/source_archives/external_quant_data/minute_repair/`. The cross-source
resolution artifact classifies every remaining excluded stock-day: 35,275 are
exact three-source minute agreements against a conflicting daily aggregate,
10,698 need mixed-source arbitration, 8,175 lack complete source coverage, 950
remain purchased-source clear candidates, and six remain Tushare clear
candidates. Exact three-source agreements are skipped by default instead of
being repeatedly downloaded or blindly overwritten.

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

The active minute research pipeline is `quantlab.research.minute_v2`. It uses
causal features at every eligible minute, next-bar entry, ordinary-share T+1,
field-level QDP quality masks and physically separate outcome labels. The
retained `/2` correctness benchmark is a one-day development slice under
`data/research/minute_v2_bench_narrow_rev5/`, not a full-month training set. Its
support query covers 62,853 eligible stock-days (3,016 symbols across 21
available trading days), while the selected output date is `2022-06-16` with
2,922 complete sessions. The base artifact has 683,748 rows (all 234 decision
minutes), and the candidate and label artifacts each have 254,417 rows.
Candidate coverage is 37.2092% in this run; that is a compute-budget
observation, not an industry standard or a strategy-validity result. The
current benchmark was rebuilt with implementation revision `2026-08-28-5`.
The previous `data/research/minute_v2_bench_narrow/` directory is retained as
an explicitly historical revision-3 artifact and is not mixed with this one.

The minute labels have distinct time semantics. `label_*m` advances N positions
on the decision grid, so it is N decision steps rather than necessarily N
elapsed trading minutes and can cross lunch or overnight. Legacy execution
fields use the next raw one-minute VWAP, while this decision-grid family uses
the next decision bar. The
`label_session_*m` family advances over raw bars within one trade date and marks
gapped or incomplete windows explicitly. The benchmark's labels are stored for
candidate keys only. The separate stage-one artifact now covers all 683,748
base keys at
`data/research/minute_v2_bench_narrow_rev5/audits/stage_one/date=2022-06-16/complete_outcomes.parquet`;
582,485 rows have an executable observed net-return label. Its gate report uses
exact within-minute, same-density random baselines and separates “at least one
top-K hit in a minute” from true top-K row recall. On this single date the gate
retains 69.06% of top-five net-return rows and 50.69% of positive-return
magnitude, but also 49.58% of negative-return magnitude. It is therefore an
attention/volatility prefilter, not a demonstrated buy signal. The replay
output remains a comparison harness that uses future exit-capacity/label fields
and entry-cost marking, not a fill-accurate inventory backtest.

```powershell
$env:PYTHONPATH='H:\quant_project\src'

# Recheck the current one-day correctness benchmark.
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quantlab.research.minute_v2 `
  --workspace-root H:/quant_project verify-month `
  --manifest H:/quant_project/data/research/minute_v2_bench_narrow_rev5/months/year=2022/month=06/manifest.json

# Reuse or rebuild the separate full-base outcomes, then refresh all recall reports.
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quantlab.research.minute_v2 `
  --workspace-root H:/quant_project stage-one-audit `
  --manifest H:/quant_project/data/research/minute_v2_bench_narrow_rev5/months/year=2022/month=06/manifest.json
```

The historical revision-3 manifest remains available at
`data/research/minute_v2_bench_narrow/months/year=2022/month=06/manifest.json`.
Running the verifier against that path intentionally reports
`minute_v2_manifest_build_revision_mismatch:2026-08-28-3:2026-08-28-5`; do not
edit it to suppress the guard.

The older `data/research/minute_v2/` and any derived `/1` products are retained
only as historical material and must not be mixed with the `/2` benchmark.
QDP source data and repair evidence were not touched.

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
minute audit. The workflow reuses valid raw captures, caps each range at five
actual trading days (1,205 possible rows), requires an exact 241-row session,
and overlays accepted price fields without replacing volume or amount:

```powershell
$env:PYTHONPATH='H:\quant_project\src'
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quantlab data minute-repair `
  --selection-mode open `
  --minimum-relative-error 0.005 `
  --maximum-relative-error 0.01 `
  --max-trading-days 5 `
  --workspace-root 'H:\quant_project'
```

Omit `--apply` for candidate evaluation. Add it only after reviewing the
generated `result.json`; valid raw captures make the command resumable. A
local Parquet candidate run uses the same command with `--local-minute-root`;
it never updates daily data and high/low installation additionally requires
independent extreme-timing evidence.

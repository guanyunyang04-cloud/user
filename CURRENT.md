# Current project state

Updated: 2026-08-30

## Decision

The project is a personal A-share research workspace. The near-term question is
whether simple, causal strategies can survive a single held-out year after
development. Use 2022-2024 for development, fitting, and strategy selection;
use 2025 once for validation. Do not add a second fold or early stopping unless
the research design changes explicitly. Hand-designed strategies and a real
inventory backtest are the preferred next path; ML baselines remain optional
comparators.

No live or paper trading is active. Existing replay outputs are not execution
evidence.

## Active surfaces

- Code: `src/quantlab/data` and `src/quantlab/research`.
- Active QDP manifest: `data/qdp/qdp_v2/active/active.json`.
- Daily research contract: `data/research/daily/manifest.json`.
- Minute-v2 development artifact:
  `data/research/minute_v2_dev_rev6_partition/`.
- Retained compatibility benchmark:
  `data/research/minute_v2_bench_narrow_rev5/`.
- Retired minute-v2 evidence:
  `data/research/archive/minute_v2_legacy_20260830/`.
- Minute-MA research contract and representative validation:
  `research/records/minute_ma_v1/`.
- Durable study inputs and evidence: `research/studies/` and
  `research/records/`.
- Local model outputs: `runs/`.

The former `daily_research/` and `quant_data_platform/` trees are retired.
Historical evidence may contain their names or absolute paths; those strings
are provenance, not active imports. The current working tree contains the
intentional refactor changes from this takeover. Do not reset or discard them.

## Refactor status

Completed in the current working tree:

- Root and research CLIs lazily import data and ML-heavy modules. Importing
  `quantlab.cli` does not load Torch or LightGBM.
- LightGBM and Torch are optional `research` dependencies. Provider-only
  dependencies are in the `sources` extra.
- JSON safety, strict/optional reads, hashes, atomic copies, and workspace
  resolution are centralized in `quantlab.core`. `QUANTLAB_ROOT` is preferred;
  `QDP_WORKSPACE_ROOT` remains a compatibility fallback.
- Read-only active-domain parsing is separate from mutating repair code.
  Repair errors, mutation, validation, and installation have explicit owners.
- Package facades no longer import an entire implementation graph. The custom
  AST import scan currently finds no cycles.
- Minute-v2 artifact fingerprints are shared by builder, training, and
  stage-one checks. The month builder is a short coordinator around focused
  preparation, materialization, and finalization stages.
- The unreachable `minute_v2/mining.py` path and obsolete repair model facade
  were removed; no active CLI or test references them.
- Runtime archive verification tolerates manifests written under the retired
  root by resolving missing archive and ledger paths beside the manifest.
- Study specifications now use workspace-relative paths where targets still
  exist. Deleted historical inputs are explicit `legacy://` provenance
  references. Rerun outputs are isolated under `runs/studies/` and never target
  durable records.

The QDP physical namespace still uses `qdp_v2`, hashed dataset generations, and
an `active.json` pointer. Flattening it is maintenance work, not a prerequisite
for minute research. Preserve atomic installation and workspace-relative
provenance if that migration is undertaken.

## Verification

The latest full run at the current source revision passed:

- the full pytest suite;
- repository Ruff checks;
- bytecode compilation;
- root-CLI lazy-import check (`torch_loaded=False`,
  `lightgbm_loaded=False`).

The final active-data checks also passed: QDP status with file verification,
QDP quick check, the manifest-first GC dry run (`unreferenced=0`), and the
current minute-v2 month verifier.

The active QDP quick/physical checks and the minute-v2 month verifier should be
run again after any data migration. Do not treat a stale historical manifest
revision mismatch as corruption.

## Data state

The active QDP cutoff is 2026-07-21. The canonical one-minute store covers
2010-01-04 through the available 2026-08-21 tail, with 3,301,495,126
continuous bars and 13,756,395 separate 09:30 rows. Source archives and repair
evidence under `data/qdp/source_archives/` are immutable inputs and must be
kept.

Historical minute quality is field-level. The final audit has 55,104 price-
excluded stock-days plus 125 independent session exclusions; the masks remain
explicit and are not a reason to discard the whole minute table. Tushare-
compatible and purchased-local sources are repair/cross-check sources only.

The active daily research manifest contains 4,191,476 stock-days through its
declared 2025-12-31 outcome boundary and 183 validated input fields. Its older
five-fold model conclusions remain historical context, not the minute-v2
acceptance criterion.

## Minute-v2 state

The current builder revision is `2026-08-29-1`. It evaluates causal bars from
09:31-11:29 and 13:01-14:55, with 234 decision groups per complete day. The
input contract includes point-in-time membership, status, industry, factors,
auction, capital, valuation, corporate actions, and minute quality masks.

The verified development month is a one-day selection (`2022-06-16`) from a
21-day support range:

- 683,748 base rows and 683,748 optional-feature rows;
- 254,417 event rows and 254,417 candidate-label rows;
- 82 core plus 97 optional model columns in `split` storage;
- all 234 decision groups present;
- separate complete-base stage-one outcomes cover all 683,748 keys.

`label_*m` advances N positions on the decision grid and may cross lunch or
overnight. `label_session_*m` stays within a trade date and marks incomplete or
gapped windows. High/low masks affect MFE/MAE independently of close returns.
The candidate gate is a volatility/attention prefilter, not a directional
signal: on the retained single date it kept 69.06% of actual top-five rows,
captured 50.69% of positive-return magnitude, and also captured 49.58% of
negative-return magnitude. It missed 57.99% of positive rows. Do not tune it to
this date or make it a hard buy universe.

The current replay still uses future exit-capacity/label fields as a comparison
harness and marks open positions at entry cost. A real held-position path,
independent market marks, and order-state reconciliation are required before
any paper-trading interpretation.

## Minute-MA event layer

The causal 60-minute event module is `quantlab.research.minute_ma`. It is kept
separate from the existing minute-v2 MA5/MA30 contract and uses MA10/20/40/60/
120/240. It builds the four session buckets 09:31-10:30, 10:31-11:30,
13:01-14:00, and 14:01-15:00. A live MA uses the current minute close; the
fixed causal intersection is the price at which that live MA would meet the
current price. End-of-hour values are retained only as explicitly marked
diagnostics, including `posthoc_catchup`.

The representative check covered 12 dates in 2022-2024 and eight fixed stocks:
96/96 target stock-days had 240 minutes, all 138,240 state rows had complete
60-minute groups, and 424 diagnostic events were generated. All 30 recorded
checks passed. This validates data definitions and reproducibility only; it is
not a strategy or profitability result, and 2025 was not read.

## Minute-MA rule study

The finite rule registry is `src/quantlab/research/minute_ma_strategies.py`.
It keeps S0 controls, causal S1 touch/reclaim variants, S2 slope/stack/first-
touch filters, and an S4 volume-ratio proxy as separate versioned rules. The
posthoc catch-up rule is diagnostic-only; S3 market and sector gates are listed
but unavailable until point-in-time breadth fields are joined. Forward outcomes
and compact statistics live in
`src/quantlab/research/minute_ma_event_study.py`; they use next-minute-open
entry, explicit percentage costs, missingness flags, and no inventory replay.

The representative strategy pilot is recorded in
`research/records/minute_ma_v2_pilot/`. Its eight-symbol/12-date scope is a
development-chain check, not evidence of full-universe profitability.

## Artifact policy

Retain the rev5 compatibility benchmark as historical evidence and keep it
separate from the current partitioned development artifact. The older
full-month product and failed `minute_v2_dev_rev6*` attempts were retired on
2026-08-30. Their original manifest, pilot audit, checkpoints, and one query
profile are in `data/research/archive/minute_v2_legacy_20260830/`; none of those
files are active inputs. Never edit an old manifest's implementation revision
to silence a guard.

Use `quantlab data gc --with-size --json` as the first step before removing QDP
generations. On 2026-08-30 the manifest-first scan removed six unreferenced
generations (639,371,865 bytes, about 0.595 GiB); a post-cleanup dry run reports
no unreferenced generations. Active datasets, source archives, and referenced
historical generations remain outside that cleanup set. Runtime workflow
captures are intentionally classified separately because several contain
recoverable archives or provenance needed for audit.

## Next actions

1. Keep QDP status/quick checks and the minute-v2 month verifier as regression
   checks after any data or code change.
2. Turn the independent operator hypotheses and minute-MA event types into
   competing hand-designed rules. Keep every attempted version and use only
   causal minute fields for entries and exits.
3. Replay the candidate rules on 2022-2024, then inspect 2025 once after the
   strategy is frozen. Report costs, missingness, recall, turnover, and drawdown
   by year and market regime.
4. Add optional tree/sequence baselines only as comparators after the hand-
   designed rules have a stable event contract.
5. Replace the comparison replay with an inventory/order-state backtest before
   considering a broker adapter. QMT integration remains deferred.

## Useful commands

```powershell
$env:PYTHONPATH='H:\quant_project\src'
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quantlab data status --verify-files
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quantlab data check --quick --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quantlab research verify
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest -q
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m ruff check src tests
```

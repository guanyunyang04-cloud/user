# Seq100 Research-Generation Orchestration Implementation

Date: `2026-07-11`

Status: `implemented and unit-tested; no model training or outer audit started`.

## Scope

`daily_research/path_policy/seq100_research_generation.py` implements the experiment state machine required by `seq100_pit_adjusted_global_tail_contract_20260711` without modifying QDP active pointers or active execution artifacts.

The registered inner experiment is fixed to purged OOS folds `2018, 2019, 2020, 2021`. It reuses `seq100_walkforward build-fold` and the three `seq100_mainline` profile commands:

- `baseline`: daily-only summary-v2 low-VA profile with smooth path-value gradient and local chunk ranking.
- `hard_st`: the same profile with hard-max forward path-value semantics and straight-through gradient.
- `hard_st_global_tail`: hard-ST plus separated `global_tail_512` ranking slates.

Every generated training command uses `fixed_oos`, final-epoch checkpointing, zero early-stopping patience, and no prediction CSV. The orchestrator generates commands and evidence registries; it does not launch model training itself.

## Selection stages

1. `init` writes an immutable candidate registry for the lightweight screen.
2. `register-result` binds metrics or a completed training run to an exact registered profile/year/seed job. Real run imports validate run tag, seed, fold year, purged view, evaluation/checkpoint policy, hard-ST mode, and ranking mode.
3. `advance-confirmation` requires the complete screen matrix and creates an immutable multi-seed confirmation registry for baseline plus the guardrail-eligible shortlist.
4. `freeze` requires the complete confirmation matrix and an explicit champion matching the registered selection policy. The immutable freeze binds all confirmation result digests.
5. `begin-outer-audit` accepts only an intact frozen champion and atomically claims the historical outer years `2022-2025`. A second claim is rejected even if the first audit is incomplete.
6. `finalize-outer-audit` requires the complete claimed matrix and records evidence grade `selection_aware_historical_oos`.

## Decision policy

The primary metric is fold-and-seed mean daily Top3 `opportunity_value` alpha. Tie-breakers are worst-fold Top3 opportunity alpha, Top3 realized-plan alpha, and daily rank IC.

Non-baseline candidates must pass guardrails relative to the baseline for:

- Top3 realized-plan alpha;
- oracle regret;
- Top10 opportunity alpha;
- daily rank IC;
- worst-fold Top3 opportunity alpha;
- fill rate;
- path MAE.

All required metrics must be finite. The baseline remains eligible, so a candidate with higher primary alpha but materially worse execution/path guardrails cannot become champion.

## Immutable artifacts

- `candidate_registry.json`
- `confirmation_registry.json`
- `frozen_champion.json`
- `outer_audit_consumption.json`
- `outer_audit_result.json`

`result_ledger.json` is the only append/update surface. A registered job result is idempotent only when its evidence digest is identical; conflicting replacement is rejected.

## Verification

Focused command:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest daily_research/path_policy/tests/test_seq100_research_generation.py -q
```

Result: `6 passed`.

Covered gates include the frozen inner matrix, immutable registry mismatch, screen/confirmation selection, baseline-relative guardrail rejection, explicit champion mismatch, run provenance mismatch, freeze tampering, single-consumption outer audit, and selection-aware outer result metadata.

## Current evidence boundary

No training has been launched by this implementation. There is no selected or frozen champion, no consumed historical outer audit, and no claim that hard-ST or global-tail improves performance. Active execution and QDP active state remain unchanged.

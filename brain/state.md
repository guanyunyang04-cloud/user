# Current state

Updated: 2026-07-30

## Objective

Build a daily A-share main-board selector that keeps upside opportunity, path
quality, and pre-peak adversity separate. Entry-label research and the
LightGBM MFE capacity closeout are complete enough to freeze a strict OOS
contract. The existing matched-capacity post-entry challenge was run against
the superseded v2 MFE contract and must be revalidated before it is used to
decide the next holding-model stage.

## Frozen entry contract

The bounded feature-union audit trained 18 stage-one boosters and read no 2026
data. No union passed its preregistered role, FDR, and path guardrails; neither
negative control passed and no three-family follow-up was eligible.

- Keep base plus `turnover_cost_proxy` for `mfe_10`.
- Keep base plus `breakout_retest_levels` for `mfe_20`.
- Extra week/month, swing, and traditional-indicator families do not enter the
  final heads.

`seq100_entry_contract_oos_v3` is the active candidate-aligned 2020-2025
stacking contract. It contains 4,337,640 rows and five output blocks:
`mfe_10`, `mfe_20`, raw three-class `state_10`, `pre_peak_mae_10`, and
`pre_peak_mae_20`.

- v3 changes only `mfe_10`, whose final head now uses a fixed 256-tree policy
  for every model year. D20 retains the existing base-head-prior-year capacity
  policy.
- Candidate row keys and the other six physical columns are byte-for-byte
  unchanged from v2. No state or risk booster was retrained.
- v2 remains a preserved historical contract, not the active contract.

- The rolling state atlases pass the formal stability gate. Adjacent ARI values
  are `0.648/0.643/0.711`; centroid correlations are `0.983/0.981/0.999`.
- Strict state predictions differ materially from the old discovery models, so
  the new contract is authoritative. Risk predictions remain highly correlated.
- Raw state outputs remain relative state scores/ranks unless a future
  use-specific calibration study supports literal probabilities.

Retained results:

- `daily_research/research_records/seq100/seq100_mfe_feature_union_audit_v1/result.json`
- `daily_research/research_records/seq100/seq100_entry_contract_oos_v2/result.json`
- `daily_research/research_records/seq100/seq100_entry_contract_oos_v3/result.json`

## Matched-capacity post-entry result

`seq100_post_entry_ab_v1` reconstructed all filled-entry landmarks without a
survivor filter: 4.321M D1, 4.315M D3, and 4.309M D5 rows. It completed all 63
boosters using identical B-complete A/B support, inner-A tree counts, a 10-day
label purge, and 2023-2025 decision folds. No 2026 row or outcome was read.

Primary `pre_peak_mae_10` challenger B failed at every age:

- D1 annual Rank-IC deltas were `-0.00049/+0.00104/-0.00021`.
- D3 deltas were `-0.00005/+0.00302/-0.00174`.
- D5 deltas were `+0.00065/+0.00311/-0.00155`.
- Only the D3/D5 2024 gains survived the nine-test BH correction. Improvements
  did not persist into 2025 or across all-sample and entry-top-5% strata.

Secondary `state_10` challenger B also failed every age. Its worst annual
ordinal-IC deltas were `-0.00692` at D1, `-0.02090` at D3, and `-0.01961` at
D5. Transported validation-year temperatures worsened Brier and log loss in all
age/variant test sequences, so post-entry state outputs are relative scores, not
stable calibrated probabilities.

The earlier no-training audit was still useful: realized paths show localized
risk associations, especially in 2024 and in some deep-adverse or strong-entry
diagnostics. The formal challenge shows that these associations do not deliver
stable incremental OOS prediction beyond daily recomputation at matched model
capacity under v2. Because v3 materially changes D10 candidate selection, this
result is retained as v2-only evidence and is not automatically transported to
the active contract.

Retained result:

- `daily_research/research_records/seq100/seq100_post_entry_ab_v1/result.json`

## MFE capacity stability result

`seq100_mfe_capacity_stability_audit_v1` tested whether one fixed tree count
could replace the volatile prior-year early-stopping counts for the two frozen
MFE heads. Capacity was selected only from complete no-early-stopping curves at
the 2019-2022 rolling origins; 2023-2025 remained decision folds. Eight curve
boosters and six fixed-capacity outer boosters completed, and no 2026 outcome
was read.

- The four-origin one-standard-error rule selected 32 trees for
  `mfe_10 + turnover_cost_proxy` and 16 trees for
  `mfe_20 + breakout_retest_levels`.
- Both fixed capacities failed the preregistered 2023-2025 adoption gates.
  D10 Rank-IC deltas were `-0.00592/-0.03989/-0.01704`; Top-5 MFE deltas were
  `-0.00096/-0.00675/-0.00436`.
- D20 improved in 2023 when 16 trees replaced the old 8-tree model, but failed
  in 2024-2025. Rank-IC deltas were `+0.01233/-0.03694/-0.03481`; Top-5 MFE
  deltas were `+0.00357/-0.01825/-0.00631`, and the endpoint-return path
  guardrail failed.
- Full no-early-stopping minima already varied materially across 2019-2022:
  D10 `23/152/43/86`, D20 `4/15/79/41`. The instability is therefore not
  explained solely by the patience value.

That study's decision was not to adopt its 32/16-tree global capacities or
rebuild v2. It rejected those particular fixed counts; it did not prove that
the baseline-head, immediately-prior-year rule was optimal.

Retained result:

- `daily_research/research_records/seq100/seq100_mfe_capacity_stability_audit_v1/result.json`

## Final-head capacity closeout

`seq100_mfe_final_head_capacity_audit_v1` compared the current baseline-head
prior-year counts with final-head prior-year tuning, fixed 256 trees, and fixed
512 trees. It trained four new tuning boosters, six shared-prefix long outer
boosters, and three conditional 2020-2022 D10 contract boosters. All six
current-policy prefixes reproduced the old predictions exactly, and no 2026
row or outcome was read.

- Final-head prior-year counts were D10 `86/241/728` and D20 `41/48/400` for
  2023-2025. All searches completed a full 100-round no-improvement interval;
  none was unresolved.
- D10 fixed 256 passed every formal gate. Its Top-5 MFE deltas were
  `+0.00125/+0.00051/-0.00009`, the six-test BH-adjusted q-value was `0.0187`,
  its worst Rank-IC delta was `-0.00153`, and MAE improved in all three years.
- D10 fixed 512 failed because tail-hit lift declined in two years despite
  improving Top-5 MFE in all three. Final-head prior-year tuning also failed.
- All D20 challengers failed. Fixed 512 improved Top-5 MFE and Rank IC in all
  three years but exceeded the preregistered MAE harm limits; fixed 256 and
  final-head prior-year tuning were less stable.
- Fixed 256 materially changed D10 candidate selection in 2023 and 2025, so
  v3 was required. Its D10 Top-5 Jaccard versus v2 was
  `0.716/0.913/0.755` in 2023-2025.

LightGBM MFE capacity research is now closed. The deployment policy is D10
fixed 256 and D20 current base-head-prior-year.

Retained result:

- `daily_research/research_records/seq100/seq100_mfe_final_head_capacity_audit_v1/result.json`

## Current decision

Use `seq100_entry_contract_oos_v3` as the active five-block entry contract.
Do not yet transport the v2-only post-entry rejection to v3: before future
holding research, rerun or otherwise formally revalidate the D1/D3/D5 A/B
challenge against the changed D10 coordinate. Until that revalidation, daily
recomputation remains the operationally simplest baseline, not a newly proven
v3 holding-model verdict.

No score fusion, exit threshold, switching value, cost buffer, holding period,
slot count, leverage, stop loss, account policy, or reinforcement-learning
policy has been selected. The planned hold-versus-switch value study is not
entered. There is no active training process. D10 uses fixed 256 trees; D20
retains its existing per-year adaptive counts. Do not reopen LightGBM MFE
capacity or train deep/reinforcement models without an explicit new decision.

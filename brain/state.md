# Current state

Updated: 2026-07-29

## Objective

Build a daily A-share main-board selector that keeps upside opportunity, path
quality, and pre-peak adversity separate. Entry-label research is complete
enough to freeze a strict OOS contract. A matched-capacity post-entry challenge
has now tested, and rejected, the need for a separate path-update model under
the preregistered D1/D3/D5 design.

## Frozen entry contract

The bounded feature-union audit trained 18 stage-one boosters and read no 2026
data. No union passed its preregistered role, FDR, and path guardrails; neither
negative control passed and no three-family follow-up was eligible.

- Keep base plus `turnover_cost_proxy` for `mfe_10`.
- Keep base plus `breakout_retest_levels` for `mfe_20`.
- Extra week/month, swing, and traditional-indicator families do not enter the
  final heads.

`seq100_entry_contract_oos_v2` rebuilt the candidate-aligned 2020-2025 stacking
contract without outer-year early stopping. It contains 4,337,640 rows and five
output blocks: `mfe_10`, `mfe_20`, raw three-class `state_10`,
`pre_peak_mae_10`, and `pre_peak_mae_20`.

- The rolling state atlases pass the formal stability gate. Adjacent ARI values
  are `0.648/0.643/0.711`; centroid correlations are `0.983/0.981/0.999`.
- Strict state predictions differ materially from the old discovery models, so
  the new contract is authoritative. Risk predictions remain highly correlated.
- Raw state outputs remain relative state scores/ranks unless a future
  use-specific calibration study supports literal probabilities.

Retained results:

- `daily_research/research_records/seq100/seq100_mfe_feature_union_audit_v1/result.json`
- `daily_research/research_records/seq100/seq100_entry_contract_oos_v2/result.json`

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
capacity.

Retained result:

- `daily_research/research_records/seq100/seq100_post_entry_ab_v1/result.json`

## Current decision

Do not build a general independent post-entry model from the tested entry-memory
and realized-path blocks. Recompute the frozen five-block entry contract daily
for both held and unheld stocks. This result does not claim that path information
can never help; it rejects this specific supervised D1/D3/D5 challenger on the
approved stability criteria.

No score fusion, exit threshold, switching value, cost buffer, holding period,
slot count, leverage, stop loss, account policy, or reinforcement-learning
policy has been selected. The planned hold-versus-switch value study is not
entered because primary risk B did not pass. There is no active training process;
the next research direction requires an explicit new user decision.

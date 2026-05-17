# alpha_path20_sequence_policy_v3 Diagnostic Note 2026-05-17

## Facts
- Full-year matrix tag: `path20_v3_full_year_matrix_20260517_01`.
- Source lake id: `policy_input_bundle__0f116a9b78c92ff045a6853d`.
- Matrix status: `completed`.
- Matrix verdict before diagnostic upgrade: `contract_passed`.
- `sequence_gru` exact replay:
  - train mean total return: `0.03556211066997217`
  - validation 2022 mean total return: `-0.006352961697067561`
  - test 2024 mean total return: `0.01745214759143421`
- `decision_transformer` exact replay:
  - train mean total return: `-0.3401041390053562`
  - validation 2022 mean total return: `-0.018361379152965873`
  - test 2024 mean total return: `0.0016569932695309486`
- Projection parity passed for both model families.
- `promotion_allowed=false`; this is not live/default evidence.

## Interpretation
- The v3 evidence loop is operational, but the strategy has not proved validation alpha.
- 2024 positive test numbers must not be interpreted as success because 2022 validation is negative.
- Decision Transformer shows low projected gross exposure in validation/test, so exposure collapse is a likely diagnostic target.
- GRU has stronger test return but still failed validation, so generalization and 2022 regime behavior need diagnosis before any model promotion discussion.

## Required Next Diagnostics
- Add replay attribution for gross return, cost, net return, benchmark return, excess return, cash, exposure, turnover, and projection distance.
- Add matrix-level `evidence_diagnostics` with warnings for under-exposure, projection overcorrection, surrogate/exact gap, and validation generalization failure.
- Preserve old evidence fields while adding diagnostics; old summaries remain readable.
- Do not change reward, loss, or model architecture until the diagnostic output identifies the dominant failure mode.

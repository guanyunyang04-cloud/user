# Seq100 PIT-Adjusted Global-Tail Research Contract

Date: `2026-07-11`

Status: `approved implementation / research only / active execution frozen`.

This contract freezes the implementation boundary for the next seq100 research generation. The machine-readable source of truth is `seq100_pit_adjusted_global_tail_contract_20260711.json` in the same directory.

## Frozen decisions

- The universe is the point-in-time Shanghai and Shenzhen A-share main board universe. Signal-day ST, delisted, suspended, ChiNext, STAR, and BSE securities are excluded. A future ST or delisting event must not remove an earlier eligible observation.
- Ranking is produced on the signal day before next-open execution facts are known. Next-open suspension or a tick-rounded open at the up-limit produces an unfilled order and retained cash; it must not remove the candidate before ranking.
- OHLC paths use the registered positive back-adjust factor; volume and amount do not use the price factor. Suspended holding days carry the last adjusted close for valuation and remain non-tradable.
- Price-label, VA-auxiliary, observation, and tradability validity are separate masks. Sparse derived-feature undefined states use explicit availability indicators.
- The model remains path-first: predicted path -> deterministic opportunity value -> daily cross-section ranking. `path_value_v2_hard_st` is the first value candidate. Realized value at the predicted exit day and oracle regret are mandatory diagnostics.
- The first ranking candidate is `global_tail_512`, with path reconstruction and ranking batches separated. Exact daily listwise VJP is conditional on the screening result.
- Profile selection uses purged 2018-2021 inner folds. A frozen champion may be audited once on 2022-2025, which remains selection-aware historical OOS.

## Activation boundary

New QDP and research artifacts use new IDs. Legacy views and completed evidence are not overwritten. No QDP pointer is activated before quality checks pass, and no active execution artifact is created or changed by this program.


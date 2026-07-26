# Seq100 exit-rule comparison — 2026-07-26

## Scope

Read-only. No training, no protected asset written, no frozen artifact touched.
Evidence scripts under ignored output: `tmp/seq100_exit_rule_probe.py` (model
picks) and `tmp/seq100_exit_rule_control.py` (random-pick control).

## Method

- Picks: daily Top-1% by the frozen LightGBM `selection_score` from the closed
  study's three completed folds (2023/2024/2025, seed 7). Control arm samples the
  same per-day count uniformly at random from the same scored universe.
- Triggers evaluated on the persisted back-adjusted forward panel
  `pack/labels/future_ohlcva_path_shards`, shape `[4017, 3419, 60, 6]`, fields
  `open/high/low/close/volume/amount`, values relative to the signal-day close.
  Rebased to the realized day-1 open so a trigger is measured from the entry.
- Execution reuses `seq100_exit_policy_audit`: `entry_open_raw` next-open entry,
  `exit_close_raw` exit mark, `exit_sellable` legal-exit deferral inside an
  80-day window, frozen `a_share_round_trip_cashflow_v1` costs (3 bps commission
  with CNY 5 minimum, stamp tax 10 bps before 2023-08-28 and 5 bps after,
  0.1 bps transfer, 7 bps slippage, 100-share lots), zero recovery when no
  sellable day exists.
- Growth normalized on a single-slot sequential clock:
  `mean(log(1+net)) / mean(holding_days) * 244`. Queueing when a pick arrives
  while capital is occupied is ignored.
- 22 rules: fixed 5/10/20/40/60, trailing stop 5/8/12/15 percent from the running
  peak close, five double barriers each with a 20-day and 60-day vertical, MA-5
  and MA-10 break, and a perfect-foresight best-close upper bound.

## Result 1: no adaptive exit beat the best fixed horizon

Annualized log growth on model picks. Worst fold is the decision column.

| rule | 2023 | 2024 | 2025 | min |
|---|---:|---:|---:|---:|
| ORACLE best close | 0.4882 | 0.7316 | 0.4592 | 0.4592 |
| fixed 60d | 0.0391 | 0.1386 | 0.0468 | **0.0391** |
| barrier +10/-10 v60 | 0.0371 | 0.1295 | 0.0552 | 0.0371 |
| trail 15pct | 0.0364 | 0.1193 | 0.0441 | 0.0364 |
| trail 8pct | 0.0263 | 0.0924 | 0.0175 | 0.0175 |
| fixed 20d | -0.0194 | 0.1474 | 0.0142 | -0.0194 |
| ma10 break | -0.0317 | 0.1231 | -0.0341 | -0.0341 |
| fixed 10d | -0.0493 | 0.1492 | -0.0159 | -0.0493 |
| ma5 break | -0.1049 | 0.1509 | -0.0958 | **-0.1049** |
| fixed 5d | -0.1282 | 0.1028 | -0.0416 | -0.1282 |

`fixed_60d` wins on the worst fold. Every trailing stop and every barrier is at
or below it. Tighter stops are monotonically worse: trail 15 > 12 > 8 > 5 on the
minimum, and the 20-day verticals are uniformly worse than the 60-day ones.

## Result 2: the MA-5 break is the worst rule tested

The owner's core discretionary rule, exit when the close breaks the 5-day
average, ranks last or next to last: -0.1049 worst-fold on model picks and
-0.6987 on random picks. Mean holding is 5.9-6.1 days and win rate is 36.6-43.7
percent, versus 47.7-56.5 percent for `fixed_20d` on the same picks. MA-10 break
is better than MA-5 but still negative on two folds. On this universe the 5-day
average is not an attack line; breaking it is mostly noise, and acting on it pays
the round trip repeatedly.

## Result 3: exit timing is worth more than the available selection alpha

Perfect-foresight best-close exit reaches 0.4592 worst-fold on model picks and
0.9141 on random picks, against 0.0391 for the best realizable rule. On random
picks the oracle exit alone produces 0.91/1.45/1.20, so with no selection skill
at all, perfect timing would compound faster than any selection-plus-fixed-exit
combination measured so far. Timing headroom is real and large; none of the
tested causal rules capture any of it.

## Result 4: cost asymmetry punishes short holdings

Mean net return per trade on random picks, 2023: `fixed_5d` -0.31 percent,
`fixed_20d` -0.96 percent, `fixed_60d` -3.99 percent. Per-trade loss grows with
holding length, but per-day growth still favors long holdings because the round
trip is amortized. Round-trip cost is roughly 15-25 bps depending on the stamp
tax era, so a 6-day rule pays about 10x the per-day cost drag of a 60-day rule.
That is most of why MA-5 loses.

## Result 5: the frozen model's picks are defensive, confirmed again

Under identical exits, model picks beat random picks in 2023 and 2024 and lose in
2025. With `fixed_20d`: model -0.0194 / 0.1474 / 0.0142 versus random -0.1855 /
-0.1260 / 0.2092. The model adds value in the two weak years and destroys it in
the strong year, consistent with the closed study's diagnosis.

## Consequences for the successor study

- The entry label may be defined against a fixed terminal exit. No tested
  adaptive rule earns the right to define the label, so the circular dependency
  between label and exit rule is resolved in favor of a fixed horizon.
- The horizon that survives the worst fold is 60 days, not 20. `fixed_60d`
  0.0391 versus `fixed_20d` -0.0194 on model picks, and -0.2409 versus -0.1855 on
  random picks, so the ordering is not stable across arms and needs a decision
  under the successor's own selection rule rather than the closed study's.
- Adaptive exit remains an open research lane with demonstrated headroom, but it
  must be justified by its own evidence, not assumed.

## Limits of this evidence

- Single-slot sequential clock only. No occupancy, queueing, or multi-slot
  allocation. No account-level compounding.
- Picks come from a model trained on the retired defensive target, so absolute
  levels are not a forecast of the successor's performance. Only the ordering of
  exit rules under a common selection is being claimed.
- Rules were evaluated on 2023-2025, already burned discovery years. Parameter
  values (stop percentages, barrier widths) were inspected on those years, so any
  specific parameter carries post-selection bias.
- Triggers use the daily close only. Intraday touch, limit-price queueing, and
  partial fills are not modeled.

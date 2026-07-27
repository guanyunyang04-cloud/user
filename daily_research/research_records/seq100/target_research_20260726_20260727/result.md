# Seq100 target research summary, 2026-07-26 to 2026-07-27

## Future-path map

The pre-2023 atlas covers 6,096,195 full-universe candidates; 5,929,931 have a
legal next-open anchor and a complete D1-D60 path. Independent horizon
clustering shows that the apparent state resolution changes with the observed
prefix: D20-D60 consistently prefer a stable coarse K2 split, while D5-D10 sit
near a K2/K3 boundary. Fixed K3 remains useful as a common descriptive
low/middle/high vocabulary, not as evidence that the market has exactly three
states.

Early strength is uncertain rather than meaningless. Only 33.13% of D5-high
paths later map to the strong D60 macro state, versus 54.60% for D20-high and
80.19% for D40-high. This supports separating an entry opportunity estimate
from later state updates based on the path that actually develops.

The D60 shape atlas finds three stable macro outcomes: declining (38.29%, median
D60 -12.33%), strong advancing (13.11%, +30.88%), and mild/oscillating (48.60%,
+3.22%). Each contains distinct temporal shapes, so terminal return alone loses
path information, but shape labels should be used only if they are predictable.

## Exit and account-structure diagnostics

In the three-year Top-1% comparison, no adaptive exit beat the best fixed exit.
Worst-year net results ranked fixed D60 (+3.91%), +10%/-10% barrier capped at
D60 (+3.71%), 15% trailing stop (+3.64%), fixed D20 (-1.94%), and MA5 break
(-10.49%). MA5 therefore remains a hypothesis about state management, not an
evidenced standalone exit rule.

The full-history single-slot variance drag is real, and moving from one to two
slots removes most of it. Earlier headline growth estimates were materially
contaminated by treating unresolved long suspensions as zero terminal recovery;
they must not be interpreted as delisting-driven ruin. The proposed low-turnover
and middle-momentum basket returned about +3.2% at D20 in 11 of 16 years, but it
was selected after inspecting 12 filters and was not statistically established.
It remains a hypothesis, not a baseline.

## Learnable targets

Short-horizon re-audit supports MFE as the entry-opportunity concept. D5, D10,
D20, and D40 MFE predictions all showed useful raw information, but redundancy
analysis reduced the first feature-family experiment to `mfe_10` and `mfe_20`.
D10 adds information about D6-D10 opportunity; D20 adds information about
D11-D20 opportunity. D5 is mostly covered by longer heads, while D40 is unstable
across recent years.

`state_10` retains information beyond MFE and is the core state challenger.
`state_20` is secondary. D3/D5 state probabilities improve path-category
diagnostics but do not provide stable independent entry-tail ranking, so they
move to the later post-entry update problem. D3/D5/D10/D20/D40 pre-peak adverse
movement remains a separate risk candidate.

The current conceptual structure is therefore:

```text
pre-entry features -> MFE D10/D20 opportunity magnitude -> entry candidate
realized D3/D5/D10 path -> continuation/oscillation/failure update -> hold or exit
```

This is a target-design result, not a complete trading strategy. No final score,
exit, holding period, slot count, leverage, or stop rule has been selected.

## Active feature experiment

The current experiment compares the same baseline against one added causal
feature family at a time for `mfe_10` and `mfe_20`, using the existing recent
three yearly folds and unchanged LightGBM/model metrics. It retains the planned
60 boosters. At the repository-simplification pause point, 28 tasks are complete
and 32 remain. Existing feature matrices, models, and predictions are reusable.

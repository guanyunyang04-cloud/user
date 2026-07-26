# Seq100 signal-quality target diagnosis — 2026-07-26

## Scope

- Study: `seq100_pit_signal_quality_v1`.
- Read-only diagnosis of the frozen target and primary metric using the three
  completed LightGBM formal folds. No training was started, no frozen artifact
  was modified, and no QDP dataset, PIT pack, or registered model was touched.
- Evidence source: `training/lgbm_lambdarank_multioutput/fold_{2023,2024,2025}/seed_7/attempts/attempt_001/predictions.parquet`
  under the ignored study output, 2,178,290 rows over 29 columns.

## Reconciled facts about the frozen primary metric

- `U = common_sustained_action_utility` at `seq100_signal_quality.py:967` is
  exactly `min(log1p(d5)/5, log1p(d10)/10, log1p(d20)/20)`. Costs are already
  inside `d*_net_return`; the earlier "cost charged three times" reading was
  wrong and is withdrawn.
- The `min` operator carries a systematic negative bias. Only 21.7% / 25.7% /
  30.9% of filled candidates have `U > 0` in 2023 / 2024 / 2025. Six model
  families showing negative mean `U` is therefore a property of the metric, not
  evidence that no alpha exists.
- `min` over horizon speeds is a robustness criterion, not a growth criterion.
  It rewards low path amplitude.

## What the trained model actually learned

Per-day Spearman IC, seed 7 LightGBM, averaged over test days:

| fold | IC vs frozen target | IC vs `U` | IC vs `d20` | IC vs `mdd20` | IC vs amplitude |
|---|---:|---:|---:|---:|---:|
| 2023 | 0.3718 | 0.2146 | 0.1256 | -0.5387 | -0.4993 |
| 2024 | 0.3430 | 0.2014 | 0.1016 | -0.5168 | -0.4872 |
| 2025 | 0.4304 | 0.1964 | 0.0675 | -0.6301 | -0.5495 |

The reported `IC = 0.3735` is against the frozen target, not against return. The
frozen target's own IC against `mdd20` is -0.67 to -0.73. Top-1% selections are
low-volatility, narrow-range, slow names. Annualized by x12, Top-1% versus all
candidates: 2023 +2.8% vs -7.8%, 2024 +22.7% vs +8.5%, 2025 +5.0% vs +34.0%.
Decile `d20` is inverted-U; in 2025 dec3/dec4 reach +3.29%/+3.31% while the
highest-scoring dec9 reaches only +1.09%. The signal is defensive: it wins in
bear and choppy regimes and loses badly in a bull regime.

## Oracle label-ceiling comparison

Method: rank candidates by the true future value of each candidate objective,
take the daily Top-1%, and report realized statistics. Annualized log growth
`12 * mean(g20)`:

| objective | 2023 | 2024 | 2025 | min |
|---|---:|---:|---:|---:|
| `g20` | 4.5339 | 5.1308 | 5.9942 | 4.5339 |
| `g20 - 1.0*mdd20` | 4.3708 | 4.8864 | 5.8530 | 4.3708 |
| `calmar = g20/max(mdd20,0.02)` | 3.5767 | 4.0911 | 4.9393 | 3.5767 |
| `U` (frozen primary) | 3.3375 | 3.8711 | 4.5701 | 3.3375 |
| `pareto_ordinal_v1` (frozen target) | 1.0313 | 1.7221 | 1.2767 | 1.0313 |
| realized model score | 0.0139 | 0.1734 | 0.0366 | 0.0139 |
| all eligible candidates | -0.1364 | -0.0432 | 0.2167 | — |

The frozen target's perfect-foresight ceiling is about 23% of `g20`'s. Ties at
the Top-1% cut were checked: mean tie count 1.0, max 1, so the ranking is not a
tie artifact.

## Oracle audit of this diagnosis

An independent read-only Oracle review accepted the diagnosis as evidence that
the frozen design favors defensive paths over terminal growth, and rejected it
as sufficient grounds to unfreeze the target. Three substantive objections:

1. Only the `g20` row is a true upper bound on achievable mean `g20`. A high
   oracle payoff shows alignment with growth, not learnability; a pure-noise
   target can have a large oracle payoff and zero achievable growth.
2. The `IC(objective, model score)` column is circular. The existing score is
   predicted pareto quality times predicted fill probability
   (`seq100_signal_quality.py:7912`), and pareto already contains `-mdd` and
   `-fade` (`:1131`). Rising correlation with heavier MDD penalties only
   recovers the old model's volatility component.
3. The ceiling script filters on `entry_filled`, which is future information and
   conflicts with `research_freeze.json:142`, while the formal evaluator
   (`:12850`) ranks all finite scores and treats non-fills as cash. Eligibility
   is 99.75% / 99.33% / 99.61%, but `k` changed on 57 days in 2024 and 39 days
   in 2025. The lambda grid was also inspected on the formal years, so the
   "96% of ceiling retained" figure carries post-selection bias and no interval.

## Governance constraint

Retargeting cannot proceed as `attempt_002`. The study contract forbids formal
replacement (`seq100_pit_signal_quality_v1.json:229`), the research freeze
requires stopping without target replacement or addition
(`research_freeze.json:267`), and `formal_matrix.json:736` is
`replacement_allowed: false`. A retarget requires closing the current study as
research-design-insufficient without altering its artifacts, then opening a new
versioned contract that rebinds F1 and the source pack by immutable hash and
records 2023-2025 as burned discovery years together with every inspected
objective and lambda.

## Open decisions

Not yet decided by the owner: the governance path, the replacement objective
form (`g20` with risk moved to qualification gates, versus
`g20 - lambda*max(0, mdd20 - tol)` with lambda and gates frozen first), and
whether to rerun the ceiling comparison under full-universe eligibility before
committing to a direction.

## Resume rule

Do not resume the interrupted TabM fold or any remaining formal cell while the
target is under review. The frozen matrix, target freeze, and research freeze
remain untouched and authoritative for `attempt_001`.

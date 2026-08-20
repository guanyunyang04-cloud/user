# Report source notes

## Reporting job

- Question: replace threshold-led research with a mathematically identified forecast and validation contract, while auditing whether the current evidence is trustworthy.
- Audience: technical.
- Scope: formal signal dates 2011-01-04 through 2025-12-31; 2010 is burn-in; no 2026 reads.
- Baseline: unconditional/date-prior and frozen existing model evidence.
- Decision-useful result: distinguish what is measured, what is inferred, what is invalidated by data quality, and what must be frozen before new training.
- Delivery mode: one MCP app report; no parallel HTML report.

## Required-structure mapping

- Title: `title`.
- Technical summary: `technical_summary` plus `headline_metrics`.
- Key findings with visual evidence: `fixed8`, `endpoint`, `dependence`, `st`, `labels`, and model evidence blocks.
- Scope and definitions: `scope`.
- Methodology/model specification: `model_spec` and `decision`.
- Limitations and robustness: `dependence`, `quality_table_block`, `validation`, and `annotation_conclusions`.
- Recommended next steps: `next_steps`.
- Further questions: `further_questions`.

## Chart map

| Section | Analytical question | Family / type | Fields | Supported claim | Data sufficiency and QA |
|---|---|---|---|---|---|
| Fixed 8% | Does one fixed barrier mean the same event across volatility states? | Comparison / grouped bar | volatility decile, hit rate, barrier type | Fixed 8% is scale-dependent; volatility alone is insufficient | 10 deciles x 2 definitions; zero-based percentage scale; candidate count and tail fields retained in tooltips |
| Annual barrier | Is the event probability stable through time? | Trend / line | year, hit rate, barrier type | Both fixed and scaled barriers vary materially by year | 15 annual points x 2 definitions; ordered 2011-2025; path counts and risk fields retained |
| Endpoint wealth | Does a positive arithmetic mean imply positive compound growth? | Comparison / grouped bar | year, arithmetic/log mean | Arithmetic and log-return estimands answer different questions | 15 years x 2 measures; signed zero context; counts and ES5 retained |
| ST boundary | Is the state discontinuity a plausible smooth market change? | Trend / line | trade date, ST rate, calculation type | A localized 2011-11 splice requires source reconstruction | More than 8 daily points around the boundary; two internally equivalent series; coverage counts retained |

Every chart is native to the report artifact, has an adjacent interpretation block, and uses a richer reviewed dataset than its visible encodings. Tables are reserved for exact audit lookup. The final artifact must pass `validate_artifact`; the visible MCP render is called only after validation.

## Caveats preserved outside the main reading path

- The flat +/-5% raw-price diagnostic is not official ST truth.
- HAC and moving-block intervals do not correct historical adaptive reuse by themselves.
- 2023-2025 is retrospective rolling OOS, not an untouched holdout.
- Scite Smart Citation review was unavailable because the monthly quota was exhausted.

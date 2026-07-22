# Daily Research brain

Daily Research is now a Seq100 research system, not a production frontend or
execution client.

## Stable surfaces

- Datasets: `daily_research/data/research_store/`
- Models: `daily_research/models/registry.json`
- Final evidence: `daily_research/research_records/seq100/index.json`
- Unfinished work: `daily_research/studies/`
- Core code: `daily_research/path_policy/`

Generated output is disposable. A completed experiment must move only its final
evidence and selected checkpoint into the stable surfaces, then remove its
experiment-only code, logs, predictions, and tests.

Data and evaluation semantics remain strict: PIT/as-of features, no future-based
candidate filtering, purged folds, explicit normalization cutoffs, identical
candidate/label/execution/cost material for controlled comparisons, and finite-
capital growth as the primary deployment objective.

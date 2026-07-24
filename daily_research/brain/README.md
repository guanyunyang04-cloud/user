# Daily Research brain

Daily Research is now a Seq100 research system, not a production frontend or
execution client.

## Stable surfaces

- Datasets: `daily_research/data/research_store/`
- Models: `daily_research/models/registry.json`
- Final evidence: `daily_research/research_records/seq100/index.json`
- Unfinished work: `daily_research/studies/`
- Core code: `daily_research/path_policy/`

Generated output is not a stable truth source by default. A completed experiment
must add an indexed compact conclusion. Successful checkpoints, predictions,
account jobs, runners, and logs may remain under ignored output when the study or
user retains them for reproduction; they are deleted only by an explicit cleanup
decision. Retained research checkpoints do not enter the formal model registry or
active execution merely because they remain on disk.

Data and evaluation semantics remain strict: PIT/as-of features, no future-based
candidate filtering, purged folds, explicit normalization cutoffs, identical
candidate/label/execution/cost material for controlled comparisons, and finite-
capital growth as the primary deployment objective.

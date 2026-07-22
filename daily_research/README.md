# Daily Research

Daily Research owns the downstream Seq100 research system. It has no active
frontend, broker adapter, order store, or live execution surface.

## Stable surfaces

- `data/research_store/`: protected packs, overlays, fold indexes, and imported
  source archives.
- `models/registry.json`: the only model lookup surface; every bundle is bound
  by SHA-256.
- `research_records/seq100/index.json`: compact final reports and key evidence.
- `studies/`: one contract for each unfinished experiment.
- `path_policy/`: shared pack, model, fold-contract, execution, and finite-
  capital evaluation primitives.

`output/` and `cache/` are disposable. A completed or retired experiment keeps
only its conclusion, key metrics, contract, and selected checkpoint in the
stable surfaces above. Experiment runners, predictions, logs, partial tasks,
and layout-compatibility tests are then removed.

## Current state

The selected research baseline is `structured_joint_turnover_180x35_v2`
(`L35V2`), selected on 2023–2025 double-slippage continuous-account evidence.
The 2026 evidence is confirmation only. Two studies remain paused and described
under `studies/`: the 2020–2022 fold extension/six-fold summary and the later
true-batch-1024 comparison.

## Commands

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_development status
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_development verify
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_development model structured_joint_turnover_180x35_v2 2026
```

Data and model work must preserve PIT/as-of features, explicit normalization
cutoffs, purged future dependencies, candidate membership independent of future
outcomes, and identical execution/cost material for controlled comparisons.
Finite-capital growth is the deployment objective; IC, alpha, path error, and
unit-time efficiency explain it.

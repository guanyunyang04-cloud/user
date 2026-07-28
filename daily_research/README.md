# Daily Research

Daily Research contains Seq100 datasets, experiments, models, and empirical
results. It has no live broker or order-management surface.

## Stable surfaces

- `data/research_store/`: packs, overlays, fold indexes, and imported sources.
- `models/registry.json`: model IDs, years, checkpoint paths, summaries, and
  compact metrics.
- `research_records/seq100/index.json`: retained conclusions and key evidence.
- `studies/`: ordinary scientific configurations for unfinished experiments.
- `path_policy/`: pack, model, time-split, execution, and finite-capital logic.

Resumable experiments write models, predictions, metrics, and task results under
`output/`. A completed or retired study keeps the scientific configuration,
conclusion, important metrics, and any useful model outputs. Git tracks source
and record history; research code does not maintain a parallel identity system.

## Current state

The selected historical sequence baseline is
`structured_joint_turnover_180x35_v2` (`L35V2`). Current label and feature
research is configured separately under `studies/`; see the root Brain state for
the current pause point rather than treating study files as a lifecycle index.

## Commands

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_development status
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_development verify
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_development model structured_joint_turnover_180x35_v2 2026
```

Data and model work preserves PIT/as-of features, explicit normalization
cutoffs, purged future dependencies, and candidate membership independent of
future outcomes. Controlled comparisons use the same rows, model settings, and
cost assumptions.

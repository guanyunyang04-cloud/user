# Project memory

This is a personal A-share quantitative-research repository. The long-term aim
is to discover statistically defensible and executable market regularities,
not to preserve a particular model family, indicator, label, or past project
belief.

## Ownership and paths

- `quant_data_platform/` (QDP) owns market-data acquisition, repair, metadata,
  point-in-time semantics, and the active data catalog.
- `daily_research/` owns Seq100 research datasets, features, models,
  experiments, account simulation, and scientific results.
- `quant_data_platform/data/qdp_v2/` contains current QDP datasets and audits.
- `daily_research/data/research_store/` contains downstream research packs.
- `daily_research/path_policy/` contains current path and account research.
- `daily_research/studies/` contains concise study configurations.
- `daily_research/research_records/` contains durable empirical conclusions.
- `daily_research/output/path_policy/studies/` contains recomputable study
  outputs and validation manifests.

## Runtime

Use `C:/Users/ASUS/miniconda3/envs/yolos/python.exe` from PowerShell. Adapt
parallelism and memory use to current free system resources while preserving a
reasonable reserve; do not impose an unnecessarily rigid low resource cap.

## Long-term preferences

- Ground decisions in mathematics, causal information timing, empirical
  evidence, and legal execution rather than narrative labels or inherited
  project assumptions.
- Treat simple rules, statistical models, tree models, and deep models as
  competing representations. Use the least complex one supported by evidence.
- Preserve research depth and universe coverage while keeping process ceremony
  light.
- Explain reasoning and conclusions in the conversation; do not create a
  separate report unless requested.

Brain files are concise, possibly stale handoff notes. Current user
instructions, code, data, and observed results always take precedence.

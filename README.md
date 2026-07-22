# Quant research workspace

This workspace has three active systems:

- `quant_data_platform/`: the QDP current data store and its update/check code.
- `daily_research/`: Seq100 datasets, registered models, compact evidence, and active studies.
- `brain/`: a small takeover map and protected-object registry.

Start with `brain/README.md`. Current machine state is in `brain/state.md`.

Historical frontends, execution clients, compatibility surfaces, and terminal
experiment frameworks are intentionally absent. Their useful research evidence
is retained under `daily_research/research_records/` and the archived references.

Useful checks:

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.integrity_check --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_development verify
$env:PYTHONPATH='H:\quant_project\quant_data_platform\src;H:\quant_project'
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli --workspace-root H:\quant_project check --quick
```

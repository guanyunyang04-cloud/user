# Quant research workspace

This personal workspace has two working systems:

- `quant_data_platform/`: current market datasets and their update, repair, and validation code.
- `daily_research/`: A-share research data and retained scientific results. The
  active short-horizon model path is `daily_research/technical/`; older Seq100
  experiments under `path_policy/` are historical/reference code.

Persistent cross-session memory is intentionally small:

- `brain/README.md` contains stable project facts and important paths.
- `brain/state.md` contains the current objective, pause point, and next action.

Research configuration lives in `daily_research/studies/`. Experimental outputs
live under ignored output directories, while conclusions worth retaining live in
`daily_research/research_records/`. The conversation remains the primary place
for explaining results and deciding what to do next.

Use the yolos Python environment:

```powershell
$env:PYTHONPATH='H:\quant_project\quant_data_platform\src;H:\quant_project'
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli --workspace-root H:\quant_project check --quick
```

# daily_research

`daily_research` is the workspace's production research and execution cortex. It contains the long-horizon research stacks, live execution console, maintenance tools, and the project's structured brain state.

## Layout

- `baseline/`: legacy and still-supported research and trade-plan pipelines
- `continuous_policy/`: current continuous-control policy training, evaluation, export, and protocol orchestration
- `deep_alpha/`: longer-horizon architecture and execution-policy research
- `execution/`: execution app, task runners, web console, and production update entrypoints
- `tools/`: guards, reports, maintenance utilities, and consistency checks
- `brain/`: current state, durable knowledge, governance, and episodic writeback
- `output/`, `cache/`, `archive/`: generated artifacts, hot caches, and cold storage

## Environment

The standard environment is `yolos`, defined by [environment.yml](/H:/new_tdx64/PYPlugins/user/daily_research/environment.yml:1).

Local prerequisite:
- `t0_project/tqcenter.py` is a workspace-local data dependency. It is not installed from Conda and must exist locally when using the `tq` data source.

## Common Entry Points

- Continuous policy formal protocol:
  `python daily_research/continuous_policy/run_continuous_policy_protocol.py ...`
- Execution app:
  `python daily_research/execution/run_execution_app.py run --task <task-name> -- ...`
- Execution web:
  `python daily_research/execution/run_execution_web.py`
- Workspace maintenance report:
  `python daily_research/tools/workspace_maintenance.py report`

## Verification

Run these before and after substantial changes:

```powershell
python -m compileall -q daily_research
C:\Users\ASUS\miniconda3\envs\yolos\python.exe -X utf8 daily_research\tools\project_consistency_check.py
C:\Users\ASUS\miniconda3\envs\yolos\python.exe -X utf8 daily_research\tools\doc_guard.py check
```

## Governance

- The canonical operating state lives under `daily_research/brain/`.
- `brain/brain_manifest.json` defines the shared main-brain contract used by all sub-brains.
- Generated experiment artifacts should stay under `daily_research/output/` and can be reviewed or trimmed with `workspace_maintenance.py`.

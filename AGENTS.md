# Workspace instructions

## Runtime

- Use `C:/Users/ASUS/miniconda3/envs/yolos/python.exe` for Python, modules, and pytest.
- The shell is PowerShell. Use a PowerShell here-string for multiline Python.
- Put `H:/quant_project` and, when needed, `H:/quant_project/quant_data_platform/src` on `PYTHONPATH`.

## Data safety

- Normal QDP updates, repairs, research outputs, and model training are allowed when they are part of the requested task.
- Do not recursively delete or bulk overwrite QDP datasets, `daily_research/data/research_store`, checkpoints, or predictions unless the user explicitly targets those files.
- Before a destructive file operation, resolve the exact path and keep it inside the intended workspace directory.

## Working style

- The conversation is the primary collaboration surface. Explain results, interpretation, uncertainty, and recommendations directly to the user.
- Keep implementation proportional to the task. Extend existing code before adding a framework, wrapper, registry, or compatibility layer.
- Brain files are optional memory aids, not authority. Current user instructions, code, data, and observed results take precedence.
- Put only durable project facts in `brain/README.md` and current cross-session state in `brain/state.md`.
- Tests should cover data integrity, causal semantics, recovery, and numerical behavior only when the changed implementation needs them.
- Research-specific dates, folds, labels, parameters, and evaluation rules belong in the research config and code, not in global instructions.

## Long tasks

- Run training in the foreground with one model process at a time.
- A runner must recognize completed semantic tasks and resume from the first incomplete task.
- Keep full logs on `H:` and console output concise. Use a local memory guard when the workload needs it.

## Git

- Preserve unrelated user changes and stage only files changed by the current task.
- Focused tests and `git diff --check` are sufficient before a commit unless the change genuinely needs broader validation.
- Write English commit messages matching repository style. Do not push unless the user explicitly asks.
- Do not amend, force-push, reset hard, or perform another destructive Git operation without explicit instruction.

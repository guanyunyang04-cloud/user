# Workspace Instructions

## Python

- Use `C:/Users/ASUS/miniconda3/envs/yolos/python.exe` for Python modules and tests.
- Do not use the base Conda environment or system Python unless the user explicitly asks.

## Commit And Push Requests

- Do not publish automatically. Run this workflow only when the user asks to commit, push, publish, or sync Git changes.
- Start with `git status --short --branch --untracked-files=all`; a tracked-only diff is not a complete change inventory.
- Include every new source, test, configuration, and documentation file that belongs to the requested task. Never use `git add .`, and never mix unrelated dirty paths.
- Run focused verification and `git diff --check`, inspect the exact staged paths, fetch the remote, and block on a remote-ahead or diverged branch. Never force-push, reset, or silently rebase.
- Prefer the repository helper with explicit `--expect-paths` and `--push`; it stages untracked task files, commits, pushes, and verifies the local and remote commit IDs in one resumable operation.
- A commit-and-push request is complete only after `HEAD` equals the remote-tracking branch and the final Git status has been reported.

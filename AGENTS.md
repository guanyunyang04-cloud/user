# Workspace instructions

## Working style

- Personal project. Keep things natural and efficient, with minimal constraints on the model.

## Long tasks

- Run long tasks in the foreground and delegate waiting to the tool layer.
- Use the longest supported blocking wait. If the tool layer yields before the process exits, continue waiting through the tool layer without manually polling progress.
- Do not perform periodic status checks, run unrelated work in parallel, or send progress commentary while the process is running.
- Resume model work only when the process finishes, fails, explicitly requests attention, or the user sends new instructions.

## Git

- Preserve unrelated user changes and stage only files changed by the current task.
- Focused tests and `git diff --check` are sufficient before a commit unless the change genuinely needs broader validation.
- Write English commit messages matching repository style. Do not push unless the user explicitly asks.
- Do not amend, force-push, reset hard, or perform another destructive Git operation without explicit instruction.

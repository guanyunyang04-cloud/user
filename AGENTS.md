# Workspace instructions

## Working style

- Personal project. Keep things natural and efficient, with minimal constraints on the model.

## Long tasks

- Run long tasks in the foreground, let the tool layer wait, and resume once the process finishes. This saves tokens.

## Git

- Preserve unrelated user changes and stage only files changed by the current task.
- Focused tests and `git diff --check` are sufficient before a commit unless the change genuinely needs broader validation.
- Write English commit messages matching repository style. Do not push unless the user explicitly asks.
- Do not amend, force-push, reset hard, or perform another destructive Git operation without explicit instruction.

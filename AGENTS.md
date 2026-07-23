# Workspace instructions

<!-- workspace-brain:start -->
## Project brain

本项目使用 `brain/` 保存持久项目记忆。

- 对代码修改、长任务、项目状态判断或受管对象变更，必须使用全局 `workspace-brain` Skill，运行 `detect`，读取 manifest、identity、working memory 及任务涉及的角色。
- 上下文压缩、任务恢复、agent 交接、用户改变目标或切换 child brain 后，必须重新执行上述接管；不得依赖压缩或中断前的对话记忆。
- 最终回答前重新核对当前目标、实际改动、验证结果和必要的 brain 写回。
- 当前事实写入 `state`，稳定知识写入 `knowledge`，可重复方法写入 `operations`，日期化证据写入 `references/`。
<!-- workspace-brain:end -->

## Runtime

- Use `C:/Users/ASUS/miniconda3/envs/yolos/python.exe` for Python and pytest.
- The shell is PowerShell. Use a PowerShell here-string for multiline Python.

## Protected assets

Do not delete or rewrite QDP datasets, `daily_research/data/research_store`, or
registered checkpoints unless the user explicitly targets that exact asset and
there is a verified replacement or recovery path. The canonical list is
`brain/object_registry.json`; verify it with `tools.brain.integrity_check`.

## Keep the system small

- Extend an existing core module before creating a new framework or wrapper.
- One concept has one current name, path, contract, and CLI. Backward
  compatibility is opt-in, not the default.
- An experiment starts as one contract in `daily_research/studies/` and writes
  process material only under ignored output. When it ends, retain the compact
  conclusion, key metrics, contract, and selected checkpoint; remove the runner,
  logs, predictions, failed directories, and experiment-only tests.
- Tests protect data integrity, PIT/no-future semantics, training/evaluation
  meaning, recovery, and key numerical behavior. Do not test retired wrappers or
  file layouts.
- Brain documents describe current objects and state. Historical detail belongs
  in references and must not re-enter the hot path.

## Long tasks

- Run one foreground training task per invocation unless the user asks otherwise.
- Set tool timeout to roughly 1.5–2 times expected runtime. Let a local supervisor
  and memory guard monitor it; do not poll from the model.
- Keep full logs on `H:` and console output event-only. The supervisor exits as
  soon as the child reaches a verified terminal state.

## Git

- Do not commit or push unless the user explicitly asks.
- Preserve unrelated dirty paths. Run focused tests and `git diff --check` before
  completion.

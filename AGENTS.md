# Workspace instructions

<!-- workspace-brain:start -->
## Project brain

本项目使用 `brain/` 保存持久项目记忆。

- 在项目接管、任务恢复、跨域状态判断、Brain 写回或受保护对象变更时使用全局 `workspace-brain` Skill。普通问答和同一逻辑任务内的局部代码编辑不重复执行接管。
- 一个逻辑任务通常只运行一次 `detect`。仅在上下文压缩、agent 交接、用户改变目标或切换 child brain 后重新接管；不得依赖中断前的对话记忆。
- 最终回答前重新核对当前目标、实际改动、验证结果和必要的 brain 写回。
- 当前事实写入 `state`，稳定知识写入 `knowledge`，可重复方法写入 `operations`，日期化证据写入 `references/`。
- 已有权威 research record 时，Brain 只保存当前状态或简短指针，不重复复制整份实验结果。
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
  process material only under ignored output. When it ends, always retain an
  indexed compact conclusion. Preserve successful checkpoints, predictions,
  account jobs, runners, and logs when the user or study marks them useful for
  reproduction; delete only material explicitly classified as disposable.
- Tests protect data integrity, PIT/no-future semantics, training/evaluation
  meaning, recovery, and key numerical behavior. Do not test retired wrappers or
  file layouts.
- Brain documents describe current objects and state. Historical detail belongs
  in references and must not re-enter the hot path.

## Validation scope

- Run the smallest tests that cover the changed behavior. Run the full
  path-policy suite only for shared path-policy behavior or an explicit gate.
- QDP checks are not a universal downstream research gate. Run quick checks when
  QDP code, active manifests, or data changed; run full checks only after a data
  rewrite, a deep data audit, or an explicit user request.
- Scope data validation to consumed domains. Daily-only research does not require
  historical intraday coverage validation.
- Do not repeat successful expensive checks during evidence-only closeout unless
  the closeout changes code or data covered by those checks.

## Long tasks

- Run one foreground training task per invocation unless the user asks otherwise.
- Set tool timeout to roughly 1.5–2 times expected runtime. Let a local supervisor
  and memory guard monitor it; do not poll from the model.
- Keep full logs on `H:` and console output event-only. The supervisor exits as
  soon as the child reaches a verified terminal state.

## Git

- Committing is allowed without asking once focused tests and `git diff --check`
  pass. Stage only the paths the current task changed, and write the commit
  message in English matching repo style.
- Do not push unless the user explicitly asks.
- Preserve unrelated dirty paths. Never stage or commit paths another task owns,
  and never commit files that may carry secrets.
- Amending, force-push, reset --hard, and other destructive git operations still
  require an explicit user request.

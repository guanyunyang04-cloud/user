# 工作树与脑区入口治理审计 2026-06-20

## 结论

- 当前仓库只有一个物理 worktree：`H:\quant_project`，绑定 `main`。
- 不存在需要清理的额外 worktree 目录。
- 本轮已纠偏到 `main`，并清理已合入 `main` 的本地 `codex/*` 分支。
- 仍保留一个未合入分支：`codex/daily-research-execution`。该分支没有独立 worktree，最后提交在 `2026-05-29`，内容集中在 execution / paper trading / trade plan universe 相关修复。
- `codex/daily-research-execution` 不应整分支合入，也不应自动删除；它应作为待审计归档分支保留。未来如需要 execution 修复，只能逐 commit / 逐文件审计后窄范围 cherry-pick，且必须先确认当前执行冻结、active artifact 和 promotion 边界。

## 已执行清理

- 已删除已合入 `main` 的本地分支：
  - `codex/alpha-v2-generalization-repair`
  - `codex/daily-research-8765-e2e`

## 保留分支

`codex/daily-research-execution` 仍有 4 个相对 `main` 未合入提交：

```text
+ 1a932fa0 1
+ 15467f00 Restore liquid500 trade-plan universe semantics
+ 72c923f1 Preserve liquid pool scope in trade plan universe selection
+ 10baab8d Handle future paper orders and execution bridge drift
```

主要影响面：

```text
daily_research/execution/app_service.py
daily_research/execution/paper_trading.py
daily_research/execution/production_signal.py
daily_research/execution/entrypoint_utils.py
daily_research/execution/run_trade_plan.py
daily_research/execution/run_research_candidate_trade_plan.py
daily_research/execution/tests/
daily_research/deep_alpha/export_live_panels_from_run.py
```

判断：

- 内容与当前 alpha_v2 模型研究主线无直接关系。
- 内容触及 execution / paper / trade plan，属于高边界区域。
- 分支基底较旧，整分支 diff 很大，直接 merge 会把过时执行语义带回当前主线。
- 删除会丢失潜在有用的 execution 修复线索，因此当前不删。

## 后续规则

- 本工作区默认 `main-branch-only`：代码、文档、实验和脑区写回均默认在 `main` 展开。
- 不默认创建或使用 git worktree；worktree / 非 `main` 分支只在用户明确要求时作为例外。
- 脑区管辖下的项目开展工作前，必须先使用 `workspace-brain` skill 作为入口，至少确认：
  - 当前目标属于哪个主脑或分脑；
  - 当前分支是否为 `main`；
  - 是否存在需要保护的 dirty paths；
  - 是否触及 active artifact、live/default、paper/broker、canonical 数据或 PIT/no-leakage 边界。
- 如果通用 skill 或外部流程建议建分支、建 worktree、开 PR、做 code review，但脑区规则要求 `main` 和个人项目直改，则服从脑区边界，除非用户显式授权例外。

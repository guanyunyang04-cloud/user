# Workspace Overview

这个工作区目前由两条主线组成：

- `daily_research/`
  - 日线研究、先进版机器学习主线、执行端、`deep_alpha` 表示学习研究。
- `t0_project/`
  - 通达信 T+0 / 盘中策略与强化学习实验区，和 `daily_research` 的次日开盘执行主线分开维护。
- 本轮项目评估与维护结论见 `PROJECT_REVIEW.md`，`t0_project` 的操作边界见 `t0_project/README.md`。

## 当前结论

- 执行端主线已经冻结为：`advanced_ml (ma50 baseline, lgbm) + liquid500 + next_open`
- `deep_alpha` 仍是最重要的研究主线，但还没有通过多窗口正式框架验证，不能进入执行端或 `shadow mode`
- `t0_project` 当前定位是独立实验区，不和 `daily_research` 的正式执行链路混用

## 立项以来的阶段演进

1. `2026-03-14 ~ 2026-03-18`
   - 建起 `baseline`，完成 `score + 5d`、市场状态过滤、状态内 profile、`up_low_breakout_v2` 等第一阶段收敛。
2. `2026-03-18 ~ 2026-03-20`
   - 加入 `advanced_ml`，并把执行口径统一到 `盘后信号 -> 次日开盘执行`。
   - 执行端股票池从全 A 收敛到 `liquid500`。
3. `2026-03-20 ~ 2026-03-22`
   - `deep_alpha` 进入正式研究框架：历史滚动 `liquid500`、`next_open`、多窗口 walk-forward。
   - 多个结构增强方向被验证后淘汰，当前收敛到 `patch-based masked pretraining + ranking fine-tune`。
4. `2026-03-22`
   - 开始补项目治理：清理入口重复、拆出共享流程、增加工作区巡检/清理工具、建立顶层维护说明。
5. `2026-03-23 ~ 2026-03-24`
   - 执行默认口径完成 `ma50 baseline` 与 `lgbm` 切换，并明确标准化为“日频目标更新”。
   - `daily_research/README.md` 与 `daily_research/daily_research_plan.md` 回收到“当前状态 / 当前计划”职责，完整时间线只保留在 `daily_research/research_log.md`。

## 代码与产物边界

- 需要长期维护的源码与文档：
  - `PROJECT_REVIEW.md`
  - `daily_research/baseline/`
  - `daily_research/deep_alpha/`
  - `daily_research/execution/`
  - `daily_research/*.md`
  - `t0_project/README.md`
  - `t0_project/*.py`
  - `t0_project/execution/`
- 主要是生成物、默认不应手工维护：
  - `daily_research/cache/`
  - `daily_research/output/`
  - `daily_research/execution/output/`
  - `t0_project/backtest_output/`
  - `t0_project/logs/`
  - `t0_project/ppo_tdx_tensorboard/`
  - 全局 `__pycache__/`

## 常用维护命令

文档编码检查：

```bash
python daily_research/tools/doc_guard.py check
```

工作区体检：

```bash
python daily_research/tools/workspace_maintenance.py report
```

Git 当前状态：

```bash
git status --short
```

安全清理 `__pycache__`：

```bash
python daily_research/tools/workspace_maintenance.py clean --targets pycache --apply
```

只做演练、不实际删除：

```bash
python daily_research/tools/workspace_maintenance.py clean --targets pycache,tensorboard
```

研究产物归档预演：

```bash
python daily_research/tools/workspace_maintenance.py archive
```

确认后执行归档：

```bash
python daily_research/tools/workspace_maintenance.py archive --apply
```

## 当前维护规则

- 文档统一用 UTF-8 保存，中文内容不要再用 shell 重定向直接追加。
- 研究结论优先写入 `daily_research/research_log.md`，规划写入 `daily_research/daily_research_plan.md`。
- `daily_research/README.md` 只保留当前状态与入口，`daily_research/daily_research_plan.md` 只保留当前优先级与停止规则，不再持续追加时间日志。
- 任何要接近执行端的新方案，都必须先过正式研究框架，而不是靠单窗口或烟测结果推进。
- 默认只清理可再生生成物，不直接删除研究输出、模型产物和持仓状态文件。
- 热区只放近期结果，过期的 `daily_research/output/` 与大体积哈希缓存按 `daily_research/archive_policy.json` 进入冷归档。
- Git 只跟踪源码、文档、策略文件和归档 manifest，不跟踪研究 payload 本体。

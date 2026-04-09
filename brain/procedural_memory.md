# 主脑方法记忆

## 1. 工作区级方法学
### 1.1 Brain-first 接管
- 跨项目判断先进入主脑，再进入具体分脑
- 新 agent 接手时，先读主脑 manifest，再读目标分脑 manifest
- 若 brain 信息足够，不应先从全仓源码盲扫
- 默认可执行入口：
  - `python daily_research/tools/brain_bootstrap.py --child <brain_id>`
  - 由脚本按 manifest 解出主脑优先的真实 boot order，而不是手工维护一份顺序副本

### 1.2 分脑自治
- 分脑负责自身的语义记忆、工作记忆、程序记忆、环境模型、情景记忆与行动系统
- 分脑不再依赖传统 `README` 作为 AI 入口

### 1.3 身子-脑子匹配
- 每个项目 brain 都必须显式描述 `body_map`
- brain 与 body 不匹配时，优先修 brain，而不是继续叠加口头说明

### 1.4 生产型目标函数规则
- 对生产型分脑，默认遵守“收益优先、非降级”
- “更稳但更低收益”不能自动升级成默认值
- 若用户明确改写目标函数，必须同步写入对应分脑 `working_memory.md`

### 1.5 更新顺序
- 结构变更：
  - 先改 `brain_architecture.md`
  - 再改 `brain_manifest.json`
  - 最后改具体脑模块
- 跨项目规则变更：
  - 先改主脑
  - 再改分脑

### 1.6 去冗余写作
- 当前状态写 `semantic_memory.md`
- 当前优先级写 `working_memory.md`
- 可复用方法学写 `procedural_memory.md`
- 环境基线写 `environment_model.md`
- 时间证据写 `episodic_memory.md`
- 操作链路写 `action_system.md`

### 1.7 热区维护流程
- 当 `workspace_maintenance.py report` 报出 `cache/output` 热区告警时，先做现状确认，再决定归档或清理。
- 默认顺序：
  - 先跑 `python daily_research/tools/workspace_maintenance.py report`
  - 再跑 `python daily_research/tools/workspace_maintenance.py archive --limit 20`
  - 确认候选里不包含当前活跃实验产物、最新执行产物与正在使用的比较基线后，再考虑 `--apply`
- 不要为了降体积而直接删除当前活跃 `deep_alpha` 输出、`execution/` 最新产物或仍被当前分脑引用的 artifact。
- 若归档策略本身需要调整，先改 `daily_research/archive_policy.json` 或对应分脑，再执行 `archive --apply`。
- 维护动作结束后，重新跑 `workspace_maintenance.py report`；若同时改了 brain 文档，再补跑 `doc_guard.py check`。

## 2. Gemini 协同方法
- 从 `2026-03-29` 起，整个 Gemini 协作模块暂时中止。
- `daily_research\tools\gemini_frontend.cmd` 只保留停用占位：
  - `status`
  - `close`
- 不再要求 Gemini `ask / closeout / doctor / pin / unpin / sessions / open`。
- 依赖型运行步骤不并行：
  - 若上一步负责产生产物、下一步立即读取该产物，这两步不得并行执行。
  - 如果误并行导致读到旧结果，必须在产物写完后重跑消费步骤，并把这类坑写回对应项目分脑。
- 当前只保留一个最小结论：
  - 我们尝试过 Gemini 自动化协作，但现阶段先停用；若未来重启，应重新设计而不是直接恢复旧规则。
- 已知边界：
  - Codex 不能直接接管一个可见终端窗口实时敲字读屏

## 3. 守卫规则
- 脑文件统一使用 UTF-8
- 主脑与分脑 manifest 必须互相可解析、可追踪
- 新增项目时，先补脑，再接入主脑 child_brains
- 任何脑网络调整完成后，必须跑 `doc_guard.py check`
## 3.1 Live 路径参数升级纪律
- 如果更优配置相对当前 live 路径只差一个旋钮，不能只靠叙述性记忆宣布升级完成。
- 必须先用同协议对照行证明它更好。
- 然后再检查当前 live 入口是否真的能表达这个旋钮。
- 如果 CLI 或 wrapper 还表达不了，先补那条路径，再宣布升级完成。
- 最终验证至少覆盖三层：
  - formal 对照证据
  - 产生产物内部配置
  - 下游消费输出
- 如果 producer 产物和外层 summary 或 meta 不一致，先信产物，再修 summary 写入器，并重跑 producer，保持审计链一致。

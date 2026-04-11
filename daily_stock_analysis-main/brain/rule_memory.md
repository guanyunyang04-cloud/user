# Daily Stock Analysis 规则记忆

快照日期：`2026-04-12`

## 1. 宪法级规则
- 本分脑是独立产品线，不接管 `daily_research` 的正式主线职责。
- 接管先读身份层与交接包，再进入具体 body。
- 多入口仓库必须先判边界，再执行改动。

## 2. 策略级规则
- 当前默认进入顺序：
  - `identity -> handoff packet -> semantic -> rule -> lesson -> temporal -> working -> action`
- 涉及 AI 协作资产时：
  - brain 是工作区级入口
  - `AGENTS.md / CLAUDE.md` 是 body 级协作资产
  - 两者应同步，但不能互相取代
- 变更默认顺序：
  - 先定模块边界
  - 再定验证矩阵
  - 再做实现

## 3. 经验级规则
- 多入口产品最怕的不是功能少，而是边界混乱。
- 先定“任务属于哪个入口 / 哪条链路”，比直接改代码更重要。
- Web / Desktop / API / Bot / Workflow 的改动应按入口分开叙述。

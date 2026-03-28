# Main Procedural Memory

## 1. 工作区级方法学
### 1.1 主脑优先
- 跨项目判断先进入主脑，再进入具体分脑
- 任何子脑结构升级都必须同步更新主脑索引

### 1.2 分脑自治
- 分脑负责自身语义记忆、工作记忆、程序记忆、环境模型、情景记忆与行动系统
- 分脑不再依赖传统 `README` 作为入口

### 1.3 身子-脑子匹配
- 每个项目 brain 都必须显式描述 body_map：源码目录、入口脚本、测试、配置、工具如何对应脑模块
- brain 与 body 不匹配时，优先修 brain，使其重新成为该项目的可用控制面

### 1.4 Agent 交接协议
- 新 agent 接手时，先读主脑 manifest，再读分脑 manifest
- 然后按 `semantic -> working -> procedural -> environment -> action -> episodic` 的顺序进入
- 若 brain 信息足够，agent 不应先从全仓源码盲扫

### 1.5 去冗余写作
- 当前状态写语义记忆
- 当前优先级写工作记忆
- 可复用方法学写程序记忆
- 运行环境写环境模型
- 时间顺序实验写情景记忆
- 操作细节写行动系统

## 2. Gemini 协同技能
- 常驻入口：
  - `daily_research\tools\gemini_frontend.cmd open`
  - `daily_research\tools\gemini_frontend.cmd status`
  - `daily_research\tools\gemini_frontend.cmd close`
  - `daily_research\tools\gemini_frontend.cmd sessions`
- 会话续接默认策略：
  - 优先固定 session id
  - 次选 `--resume latest`
- 已知边界：
  - Codex 可以开关与复用 Gemini 会话
  - Codex 不能直接操纵一个可见终端窗口实时敲字

## 3. 守卫规则
- 脑文件统一使用 UTF-8
- 结构变更先改 `brain_architecture.md`，再改 `brain_manifest.json`
- 下级脑重命名或新增模块时，主脑必须同步更新
- 新增项目时，先补 brain，再允许把它纳入主脑 child_brains

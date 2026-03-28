# Daily Research Procedural Memory

## 1. 作用
本文档保存 `daily_research` 分脑已经学会、以后应反复复用的方法学与协同技能。

它只记录“怎么做”，不记录按日期排列的单次实验。单次实验统一写进：

- `daily_research/brain/episodic_memory.md`

## 2. 正式研究技能
### 2.1 当前代码口径优先
- 只要底层特征、状态机、训练窗口或执行假设发生实质变化，旧 formal 结论自动降级为历史口径
- 任何接近执行端的升级讨论，都必须回到“当前代码 + 当前特征空间 + 当前 rolling ML scores”重跑

### 2.2 锚定对照优先
- 当系统发生底层漂移时，优先做 anchored comparator，而不是口头比较两轮看起来相似的高收益数字
- 默认对照框架继续固定为：
  - 历史滚动股票池
  - `next_open`
  - 多窗口 walk-forward
  - 当前 live default 为 benchmark

### 2.3 split verdict 的处理方式
- 如果 offense 与 defense 分别落在不同候选，不再强迫选出单一全局赢家
- 下一步应改写问题：
  - 从“谁是全局默认值”
  - 变成“如何把 offense edge 与 defense edge 收进同一套控制器”

### 2.4 默认值升级规则
- 只有同一 formal 口径下稳定跑赢当前 live default，才允许升级执行端
- 任何只修一个弱窗口、却破坏更强窗口的方案，一律不得晋级

## 3. 分脑写入技能
### 3.1 写入路由
- 当前稳定状态写 `semantic_memory.md`
- 项目背景与瓶颈写 `project_map.md`
- 当前优先级写 `working_memory.md`
- 环境与命令口径写 `environment_model.md`
- 执行流程写 `action_system.md`
- 单轮实验写 `episodic_memory.md`
- 可复用方法学写本文件

### 3.2 UTF-8 安全写入
- 中文文档不再用 shell 重定向直接追加
- 文档改动后默认运行：
  - `python daily_research/tools/doc_guard.py check`

### 3.3 主脑接入协议
- 新 agent 接手 `daily_research` 时，默认先读：
  - `brain/brain_manifest.json`
  - `daily_research/brain/brain_manifest.json`
- 然后再按顺序进入：
  - `semantic_memory.md`
  - `working_memory.md`
  - `procedural_memory.md`
  - `environment_model.md`
  - `action_system.md`
  - `episodic_memory.md`
- 若主脑与分脑的判断出现冲突，以主脑边界和当前分脑实际落盘状态一起校准，不允许跳过 brain 直接盲扫 body

## 4. Gemini 协同技能
### 4.1 常驻前台窗口
- Gemini 前台协同入口：
  - `daily_research/tools/gemini_frontend.cmd`
- 常用命令：
  - `daily_research\tools\gemini_frontend.cmd open`
  - `daily_research\tools\gemini_frontend.cmd status`
  - `daily_research\tools\gemini_frontend.cmd close`
  - `daily_research\tools\gemini_frontend.cmd sessions`

### 4.2 会话连续性规则
- 关闭前台窗口会终止当前交互进程
- 但 Gemini session 本身不会自动消失，可以继续 `resume`
- 如果只追求便捷，可用：
  - `--resume latest`
- 如果追求强连续性，应优先固定 session id，而不是长期依赖 `latest`

### 4.3 Codex / Gemini 分工
- Codex 负责主线判断、文件修改、结果落盘与一致性收口
- Gemini 负责交叉核对、补充视角、长文归纳和方案对照
- 最终以当前工作区实际验证与落盘结果为准

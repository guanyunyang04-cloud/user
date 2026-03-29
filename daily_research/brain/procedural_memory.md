# Daily Research Procedural Memory

## 1. 作用
本文档保存 `daily_research` 分脑已经学会、以后应反复复用的方法学与协同技能。

它只记录“怎么做”，不记录按日期排列的单次实验。单次实验统一写进：

- `daily_research/brain/episodic_memory.md`

## 2. 正式研究技能
### 2.0 利润优先非降级原则
- 用户的原始目标是“持续提高总利润”，不是“以降收益为代价换稳健”
- 任何坏市场、弱窗口、稳定性专项，都默认建立在以下前提上：
  - 不降低当前已知最强收益前沿
  - 或通过阶段切换 / 控制器方式提升总利润
- 如果某方案只是：
  - full 端收益更低
  - 但弱窗口更稳
  - 且没有证明总利润更高
  则它只能记为防守型研究结果，不能直接升为执行端默认值
- 若用户没有明确改目标函数，默认禁止把“稳定性更好”解释成“可以接受收益下降”

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
- 只有同一 formal 口径下同时满足以下条件，才允许升级执行端：
  - 不低于当前已知收益前沿
  - 且不破坏关键弱窗口 / 关键状态
- 任何只修一个弱窗口、却破坏更强窗口的方案，一律不得晋级

### 2.5 收益口径解读规则
- `full_total_return / full_excess_total_return` 是全样本总收益，不是年化收益
- `full_annual_return / full_excess_annual_return` 才是年化口径
- 窗口结果若写成 `127.99% / 2.816` 这类双值，默认应读作：
  - 前者是该窗口的总收益或超额总收益
  - 后者是 Sharpe，而不是年化收益
- 任何“旧高收益”和“当前 live 收益”的比较，必须先核对：
  - 是否同一特征空间
  - 是否同一训练窗口
  - 是否同一股票池与 `next_open` 假设
  - 是否比较的是总收益、年化收益还是 Sharpe

### 2.6 同口径 stack A/B 规则
- 当用户质疑“改完为什么不如以前”时，不再靠旧日志口头争论
- 标准动作是直接做同一 formal 长窗口下的 stack-to-stack A/B：
  - 旧 offense 栈
  - 旧 live-like 栈
  - 新 offense-like 栈
  - 新 live 栈
- 默认输出不能只给一句 delta，还要同时给三层信息：
  - 直接 stack A/B
  - profile-only delta
  - 同 profile 内部 `v250 / v255` 切换 delta
- 如果结论表现为：
  - full 端更弱
  - 但 recent / weak / focus-weak 更强
  则统一表述为“收益结构重排”，而不是草率归类成“整体升级”或“白改了”
- 当前这类 A/B 的复用脚本为：
  - `daily_research/baseline/render_market_feature_stack_ab.py`

### 2.7 历史快照复刻规则
- 如果当前代码即使锁定旧参数、旧 profile，仍无法复刻历史日志里的关键结果，就不要继续拿当前代码硬凑
- 标准动作改为：
  - 用 `git worktree` 拉起对应历史 commit 的隔离快照
  - 直接运行当时的原脚本
  - 再把输出产物和 commit 元数据拷回当前工作区
- 这样可以区分两类问题：
  - 结果本来就是旧系统特有
  - 结果只是日志口径或记忆口径误读
- 当前这轮历史高收益审计已经证明：
  - `2026-03-24` 的 `lgbm / histgb / etr` 高收益不是空话
  - 但它属于历史快照系统，不属于今天的 current live benchmark
### 2.8 Cross-profile 控制器的收官判定
- 当 offense edge 与 defense edge 已明确落在不同 feature profile 上时，不要再只靠同 profile 内部 gating 近似它们的差异。
- 标准动作是直接做 cross-profile formal comparator，而不是继续口头推演；当前复用脚本为 `daily_research/baseline/scan_cross_profile_attack_defense_controller.py`，静态控制必须同时包含当前 live-defense 锚点和当前 clean offense 锚点。
- 评估顺序固定为：先看 `full_annual_return / full_excess_sharpe`，再看 weak-window 与 focus-weak Sharpe，最后看 offense gating 占比、turnover 与 drawdown。
- 如果 best cross-profile dynamic 只是明显补强 weak / focus-weak，但仍未超过静态 offense 的 `full_annual_return` 或 `full_excess_sharpe`，则应把它记为“稳健性 frontier”，而不是“新的收益前沿”。
- 一旦在同口径 formal 下得到这个结论，就停止把当前 `v250 / v255 / controller` 参数空间当作第一研发前线，转向新机会集 / 新 alpha 家族。
### 2.9 最小矩阵长跑的恢复规则
- 对 `deep_alpha/run_minimal_matrix.py` 这类长时间 formal stage，如果 shell 等待超时或中途中断，不要换新 tag 重来；优先用同一个 `root-tag` 直接重跑。
- 这类 runner 会自动 `skip-existing`、复用已完成的 run / pretrain artifact，并在最后一块补齐后继续写出阶段 summary 与 selected recipe。
- 只有当 `stage_<name>_selected.json` 已落盘时，才把该阶段视为正式完成；仅有各窗口 `metrics.json` 还不算阶段收口。

### 2.10 Per-window cache 提速规则
- 对 `deep_alpha` 这类多窗口正式矩阵，默认把“能否复用 per-window 中间产物”当成研究设计约束，而不是跑完后才补救。
- 若改动没有触及窗口切分、股票池、lookback、targets 或 encoder / pretrain 本体，就优先复用 feature cache、sequence corpus cache 与已落盘 encoder / pretrain artifact。
- 对只改 `score_head` 或 `ranking` 的实验，不要按“从零重跑整条慢链路”来思考；先判断哪些阶段可直接复用，再决定是否值得开跑。
- 当前实测最重的时间瓶颈是每个 window 内部的 `[4/8] Building sequence features and targets` 与 `[6/8] Training deep alpha model`。
- 因此前沿抬升预期弱的小旋钮候选，应先缩小矩阵或直接降级，避免浪费长时间全流程重训。
### 2.11 新机会集下的小旋钮重开规则
- 某个小旋钮若已经在当前机会集下输掉，不代表它在所有机会集里都永久失效；但只有在机会集已经发生实质变化，且旧证据显示它可能对股票池 / 容量约束敏感时，才允许重开。
- 推荐顺序固定为：
  1. 先用当前 winner 做最小机会集对照，确认新机会集本身是否抬高前沿；
  2. 只有当新机会集确认更强后，才在该机会集下重开 1 到 2 个最有依据的小旋钮；
  3. 若重开候选只是抬高均值、却重新引入明显负窗口，则把它记为激进分支，不覆盖稳定前沿。
- 对“更激进的收益翻译方式”也遵守同样纪律：先在更强机会集上试，再判断是否值得继续；如果集中持仓在 `3` 个窗口里破坏了 `2` 个窗口，就直接降级，不继续沿这条线猛推。
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

### 3.4 决策后遇阻的因果隔离流程
- 如果 brain 已给出方向，但代码结果、回测结论或口径解释出现明显矛盾，不要立刻扩大战线；先把问题收口成一个最小因果问题。
- 先拆成两层：
  - apples-to-apples 的同口径差异
  - apples-to-oranges 的策略/配置差异
- 优先固定不变量：
  - `features.py`
  - `build_ml_target()`
  - 股票池
  - 回测命令口径
  - 输出窗口
- 然后只改一个机制，放进隔离 worktree 做 single-switch ablation，不在主工作区直接混改。
- 对 `next_open` / walk-forward / 滚动训练问题，优先检查：
  - label-safe gap
  - train_end 与 predict_start 的边界
  - 是否存在 look-ahead bias / label leakage
  - 股票池或状态标签是否跨窗泄漏
- 如果 single-switch ablation 能精确复现旧高收益，应先把旧收益视为 artifact 候选，而不是继续按“更高收益”晋级。
- 困难解决后，默认回写顺序为：
  - 可复用方法写 `procedural_memory.md`
  - 本轮证据链写 `episodic_memory.md`
  - 若结论推翻当前默认判断，再同步 `working_memory.md` 与 `action_system.md`

## 4. Gemini 协同状态
### 4.1 当前结论
- 从 `2026-03-29` 起，整个 Gemini 协作模块暂时中止。
- `daily_research/tools/gemini_frontend.cmd` 只保留停用占位用途；不再作为默认工作流的一部分。
- 不再要求 `ask / closeout / doctor / pin / unpin / sessions / open`，也不再要求 final-answer closeout。

### 4.2 保留的最小记忆
- 这轮尝试已经发生过，细节留在 `episodic_memory.md` 作为历史证据。
- 当前只保留一个高层结论：
  - Gemini 自动化协作链路比人工前台对话更脆弱，主要瓶颈在 session hygiene、后台一次一调用、超时与格式校验。
- 在模块重新启动前，Gemini 输出不参与默认决策链。

### 4.3 若未来重启
- 视为一次新的设计问题重新评估，不自动继承这轮 `session hygiene / closeout` 假设。
- 重启前先确认：
  - 目标到底是人工对话便利，还是 Codex 自动化复核
  - 如果两者目标不同，应拆成不同工具，而不是继续混成一个模块

### 4.5 Dependent runtime sequencing
- Do not parallelize producer-consumer runtime steps.
- Typical forbidden pair:
  - `update_model.py` produces `latest_ml_model.joblib`
  - `run_trade_plan.py` consumes `latest_ml_model.joblib`
- If a downstream step reads an artifact produced by the upstream step, run them sequentially and verify the downstream output reflects the new artifact timestamp.
- If parallel execution was started by mistake, rerun the consumer step after the producer finishes and record the dependency pitfall in `episodic_memory.md`.
### 4.6 Same-Protocol Bridge Validation
- When execution default and research live anchor differ mainly by one numeric knob, compare them under the exact same protocol before changing the live default.
- First lock the shared parts:
  - same stock pool logic
  - same `next_open` execution mode
  - same feature/profile stack
  - same state-ensemble weights
  - same walk-forward windows
- Then compare only the target knob rows. For the 2026-03-29 execution upgrade, the decisive bridge rows were:
  - `trend_up_low_vol_ml25_none25_v250 @ 504 / 21 / 260`
  - `trend_up_low_vol_ml25_none25_v250 @ 504 / 21 / 520`
- Decide from the metric delta table, not from narrative memory. The minimum decision set is:
  - `full_excess_total_return`
  - `full_excess_sharpe`
  - weak-window excess total return / Sharpe
  - focus-state weak-window Sharpe
  - turnover / drawdown
- If the winning knob cannot yet pass through the execution entrypoints, patch the CLI / wrapper first; otherwise the project will keep "thinking" it upgraded while the live artifact still runs the old value.
- Confirm the upgrade on three layers:
  - formal compare row
  - artifact internal `ml_config`
  - downstream consumer output such as `latest_trade_plan.txt`
- If artifact internal config and outer meta JSON disagree, trust the artifact first, patch the meta writer, and rerun the producer step so later audits do not get misled.

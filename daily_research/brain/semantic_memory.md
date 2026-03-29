# Daily Research Semantic Memory

## 1. 作用
本文件是 `daily_research` 分脑的语义记忆，负责保存长期稳定、非时间顺序的项目认知。

它回答四个问题：

- 这个分项目是什么
- 当前稳定执行主线是什么
- 这个分脑有哪些模块
- 应该按什么顺序进入其它脑模块

上级主脑位于：

- `brain/master_brain.md`
- `brain/brain_manifest.json`

当前接入状态：

- 已接入主脑
- 已被主脑纳入 `child_brains`
- 后续 agent 可先接主脑，再顺主脑进入本分脑

## 2. 项目身份
`daily_research` 是当前正式维护的日线研究与执行分项目，负责三类任务：

1. `baseline / advanced_ml` 的可解释研究、正式回测与执行复核
2. 盘后更新离线模型，生成次日开盘手工执行建议
3. `deep_alpha` 表示学习主线研究

## 3. 当前稳定主线
- 正式执行主线：
  - `advanced_ml_current_code_live_anchor (ma50 baseline, lgbm520 v250) + liquid500 + next_open`
- 调仓语义：
  - 日频目标更新，默认 `rebalance_freq=1d`
- 执行方式：
  - 盘后生成计划，次日开盘人工执行
- 当前执行后端：
  - 当前 `execution/update_model.py` 与 `execution/run_trade_plan.py` 直接走当前仓执行链路
  - 默认注入 `regime_ma_window=50`、`enhanced_profile=up_low_breakout_v2`、`trend_up_low_vol=ml:0.25,none:0.25,v2:0.50` 与 `lgbm_n_estimators=520`
- 当前研究侧对照锚点：
  - 执行默认：`expanded_v24 + trend_up_low_vol_ml25_none25_v250 @ 504 / 21 / 520`
  - 研究进攻对照：`trend_up_low_vol_ml25_none20_v255 @ 504 / 21 / 520`

## 4. 分脑模块
- `daily_research/brain/semantic_memory.md`
  - 长期稳定认知
- `daily_research/brain/brain_architecture.md`
  - 分脑内部结构与写入路由
- `daily_research/brain/project_map.md`
  - 项目背景、主线演化、瓶颈、未来方向
- `daily_research/brain/working_memory.md`
  - 当前默认决策、优先级、停止规则
- `daily_research/brain/procedural_memory.md`
  - 已验证的方法学、文档技能、以及 Gemini 协作尝试的停用结论
- `daily_research/brain/environment_model.md`
  - 解释器、依赖、命令口径与工具入口
- `daily_research/brain/action_system.md`
  - 盘后执行链路与操作流程
- `daily_research/brain/episodic_memory.md`
  - 时间顺序实验、证据与结论
- `daily_research/brain/brain_manifest.json`
  - 机器可读索引

## 5. 身子与脑子的映射
- 研究 body：
  - `daily_research/baseline/`
- 执行 body：
  - `daily_research/execution/`
- 长期研究 body：
  - `daily_research/deep_alpha/`
- 工具 body：
  - `daily_research/tools/`
- 产物 body：
  - `daily_research/cache/`
  - `daily_research/output/`
  - `daily_research/archive/`

brain 负责解释这些 body 应该如何被理解和接入。

## 6. 进入顺序
默认进入顺序：

1. 先读本文件
2. 再读 `brain_architecture.md`
3. 再读 `project_map.md`
4. 再读 `working_memory.md`
5. 需要方法时读 `procedural_memory.md`
6. 需要命令和环境时读 `environment_model.md`
7. 需要执行细节时读 `action_system.md`
8. 需要证据时读 `episodic_memory.md`

## 7. 当前结构判断
- `advanced_ml` 继续承担正式执行职责
- `deep_alpha` 继续承担长期研究职责
- 动态控制器是当前执行端下一步研发方向
- `daily_research` 仍是整个工作区的生产主线分脑

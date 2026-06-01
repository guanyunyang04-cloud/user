# Daily Research 治理层

快照日期：`2026-04-13`

## 1. 治理目标
- 防止 strongest-model、learned-control 和 live 主线发生目标偏移、上下文偏移和证据偏移
- 强制所有正式实验都走“读取状态 -> 检查 -> 执行 -> 写回 -> 重规划”闭环

## 2. 执行前四检
- 目标一致性检查
  - 这件事是否仍服务当前主问题和当前生产主线
- 规则冲突检查
  - 是否违反 formal / recent / live / promotion 分层、strict resume、前台执行等硬规则
- 经验教训检查
  - 是否踩中 `knowledge_center.md` 中已知失败模式
- 依赖完整性检查
  - 数据、脚本、解释器、预算和上下文是否足够

## 3. 默认接管纪律
- 新 agent 默认先读：
  - `identity_layer.md`
  - `state_center.md`
  - `knowledge_center.md`
  - `operations_center.md`
- `episodic_memory.md` 只在需要完整证据时再读

## 4. 写回纪律
- 当前状态、当前优先级、当前 handoff、当前时态：
  - `state_center.md`
- 稳定事实、规则、教训：
  - `knowledge_center.md`
- 新命令口径、入口变化、环境和流程：
  - `operations_center.md`
- 时间顺序过程和原始证据：
  - `episodic_memory.md`

## 5. 守卫
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.doc_guard check`
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.integrity_check --json`
- `C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/tools/project_consistency_check.py`

## 6. 执行端冻结治理
- 2026-06-01 起，执行端状态为 `frozen_skeleton_only / awaiting_research_rebuild`。
- 冻结期允许保留和维护执行代码骨架、只读诊断、数据 readiness、候选 backtest / trade-plan wrapper、active artifact guard 和恢复盘点。
- 冻结期禁止 live/default、paper/live/broker 接线、正式交易计划生产、active manifest promotion、production root 重建、自动化每日执行，以及未授权删除执行合同。
- 解冻必须满足三项前置：研究端差异解释完成；新旧模型同口径候选回测方案完成；旧 production payload 的恢复/归档/替代路径获得明确授权。
- `rebuild_lineage_diff_audit` 只能满足“研究端差异解释”的初步证据要求；不得把该审计误读为模型胜负、short_v5b 替代、执行端解冻或 active artifact 重建授权。
- 旧 multi-horizon / short_v5b 等价比较必须使用主板口径 pool；包含 `300/301/688/689` 创业板/科创板前缀的 new-lineage rebuild 只能作为 `wrong_universe_diagnostic`，不得作为 promotion、执行解冻或 short_v5b 等价对照证据。
- Corrected mainboard-only baseline 若只是 `near_pass`，仍不得作为 active promotion、short_v5b 替代或执行端解冻证据；必须先处理 hit lift min negative、旧 `156` feature schema 未恢复、short_v5b payload 缺失和同协议 bridge 未完成这四类 blocker。
- TQ vs BaoStock diff 若处于 `blocked_tq_unavailable`，不得作为数据源胜负、模型胜负、旧 Stage 2.8 失效、short_v5b 替代或执行端解冻依据。若 diff 已运行且 label 等价，也只证明该抽样/口径下 next-open label 近似一致；在旧 `156` feature schema、short_v5b payload、amount/missing/fill policy 差异未闭环前，仍不得作为执行端解冻或模型替代依据。

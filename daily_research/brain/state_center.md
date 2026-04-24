# Daily Research 状态中枢

快照日期：`2026-04-24`

## 当前结论
- `daily_research` 仍是当前工作区的正式生产研究与执行主线。
- 当前 active 执行物化真源为 `daily_research/output/active_execution_strategy.json`。
- 当前 live 默认执行 label 为 `short_expert_policy_v5b__regoff_k1_20d_ensemble_native_anchor__active`。
- 当前 effective live execution profile 为 `regoff_k1_20d_ensemble_native_anchor`。
- 当前 production root 为 `daily_research/output/short_expert_policy_v5b_execalign_production_default`。
- 当前 continuous_policy 研究主线已从 r10 deploy executability 推进到 r11/r11b sell-source、held-side release/funding 与 order translation 的耦合问题。
- r11/r11b 全部分支仍是 `shadow_only`；不得把 `result_value_v10`、`alpha_result_value_budget_split_v11` 或任何 confirm 分支直接解释成 promotion / live 切换证据。
- 本轮只是历史归档压缩；未启动训练，未运行新的行为审计，未切换 live，未改写 promotion gate。

## 当前接管入口
- 默认读取顺序仍为：`identity_layer.md -> state_center.md -> knowledge_center.md -> operations_center.md -> governance_layer.md`。
- `episodic_memory.md` 只作为过程复盘入口；历史原文和长命令默认进入 `daily_research/brain/references/`。
- 运行 `daily_research` 程序必须显式使用 `C:/Users/ASUS/miniconda3/envs/yolos/python.exe`，并设置 `PYTHONIOENCODING=utf-8`、`PYTHONUTF8=1`。
- PowerShell 出现中文乱码时，先用显式 UTF-8 复读，不得直接判定文档损坏。

## 当前研究状态
- strongest-model 主线已收口，不是当前阻塞点；learned-control 的生产执行仍以 `policy_v5b` active artifact 为准。
- continuous_policy 当前已具备训练、评估、shadow continuity、导出、行为审计与 study scoring 闭环。
- r10 证明 deploy intent 可执行性可以被度量和约束，但不能只用 clip reduction 代替 deploy executability。
- r11 证明 sell-source contract 可以进入训练侧，但 `result_value_v10` 本身尚未成为稳定预算目标。
- r11b 证明 `alpha_result_value_budget_split_v11` 能把 funding-sell 推向更少、更干净，但 `deploy_funding_release_consistent_share` 仍为 `0.0`，且 deploy/order translation 仍会互相拉扯。
- 当前真正瓶颈收敛为：`held-side release learning + order translation drift + deploy executability` 三方耦合。

## 当前优先级
- 冻结 live 默认执行，不做静默切换。
- 继续把 r11/r11b 视为 research / shadow 证据，而不是 production 证据。
- 后续若继续推进 continuous_policy，应先处理 release/funding 学习与订单翻译漂移的兼容性。
- 不回退到 simulator-only 解释，不把单一 aggregate share 当作结论；需要逐仓明细时读取 held-side detail audit。
- `analyze_behavior_gap.py` 不得并行运行多个会写 `latest_behavior_audit_summary.json` 的实例。

## 当前边界
- formal、recent、promotion、live 不得混写。
- 不足正式证据的 smoke / dry-run / short-window check 不能升级为正式 verdict。
- continuous_policy 只有在正式协议、参考对照、连续 shadow continuity、行为语义和 promotion gate 均稳定后，才允许进入 promotion 讨论。
- 当前任何 r11/r11b 结果都不改变 active execution artifact。

## 当前风险
- 若身份层再次写入具体 live 默认、最新分数或 winner，属于文档职责漂移。
- 若状态中枢继续堆叠日期段，后续接管会重新退化为长日志扫描。
- 若只看 r11b 自动 confirm，会错过 `confirm_03_semantic_v11v9` 这条重要语义对照线。
- 若并行运行行为审计，仍可能重现 latest 摘要文件竞争。

## 推荐下一步
- 若继续研究：围绕 release learning、order translation drift、deploy executability 设计联合验证，而不是单独扩大 release loss。
- 若继续维护：优先保持入口文档轻量，把过程证据写入 `episodic_memory.md` 或 `brain/references/`。
- 若需要旧状态细节：按下方索引读取历史原文，不把归档历史自动提升为当前状态。

## 历史归档入口
- 原 `state_center.md` 已原样归档：`daily_research/brain/references/state_center_history_raw_20260424.md`。
- 标题索引：`daily_research/brain/references/state_center_evidence_index_20260424.md`。
- 原始行数：`1863`。
- 原始 SHA256：`1c135d58f6e1951cf60c8bd2234522ccc5357755953f6962d9070ec67b7a1e16`。
- 读取纪律：当前状态以本文件上方章节为准；归档文件只作为历史证据与追溯入口。

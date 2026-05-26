# Alpha Multi-Horizon Utility Policy Mainline Rename - 2026-05-23

## 决策
- Status: `research mainline rename / governance / shadow-only / no strategy change`.
- User decision date: `2026-05-23`.
- Current research mainline pointer: `alpha_multi_horizon_utility_policy_v1`.
- Chinese name: `多 Horizon 交易效用排序主线`.
- Previous current pointer: `alpha_path20_neural_policy_v1`.

## 事实
- 最新 completed evidence 是 `path20_horizon_discovery_no_alpha_gru_liquid500_h1_2_3_5_8_10_15_20_30_du_cost20_hit10_dd010_20260523_01`。
- 该 study 已经不再是固定 20 日路径预测：它使用 horizon grid `1,2,3,5,8,10,15,20,30`，主 score 是 `trade_utility_score`。
- `trade_utility_score` 通过 corrected validation/test selection gate，但 predicted best horizon 在 test 中明显塌缩到 `30d`。
- `daily_research/output/active_execution_strategy.json` 不因本次命名迁移改变。

## 推断
- 继续把当前主线命名为 `Path20` 会把后续研究误导回固定 20 日路径误差。
- 更准确的当前目标是：学习多 horizon 的交易效用排序，先解决 score / horizon calibration，再谈 seeds、universe 或执行层扩展。
- 当前模型更像强的 `20-30d` utility ranking candidate，不是已经完成的逐样本 horizon chooser。

## 命名规则
- 新研究主线名固定为 `alpha_multi_horizon_utility_policy_v1`。
- 新实验 tag 默认使用 `mh_utility_...` 前缀，并显式写入 horizon grid。
- 旧 `path20_...` tag、`alpha_path20_neural_policy_v1`、`alpha_path20_sequence_policy_v1` 和 `daily_research.path_policy.run_alpha_path20_protocol` 保留为历史证据、shadow comparison 或代码 namespace。
- 不批量改写历史 reference、run tag、artifact path 或输出目录；历史名字是证据轨迹的一部分。

## 边界
- 本次是命名与治理迁移，不是 live/default、promotion、allocator、replay 或 active artifact 变更。
- 不启动 liquid800，不启动 multi-seed，不改 `daily_research/output/active_execution_strategy.json`。
- 后续研究结论必须区分：历史 Path20 evidence、当前 multi-horizon utility evidence、live active execution。

## 下一步
- 先做 constrained horizon-score / calibration variant，目标是减少 near-total `30d` collapse。
- 如果约束后仍保持 validation/test rank IC、spread、hit lift 和月度稳定性，再运行 seeds `7,11,19`。
- 如果 multi-seed 稳定，再判断是否把 broad horizon grid 简化为 `15/20/30d` longer-horizon utility family。

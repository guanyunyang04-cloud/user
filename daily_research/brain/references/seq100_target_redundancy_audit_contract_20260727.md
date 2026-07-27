# Seq100 目标冗余审计结果前合同

日期：`2026-07-27`

研究：`seq100_target_redundancy_audit_v1`

合同：`daily_research/studies/seq100_target_redundancy_audit_v1.json`

合同 SHA-256：
`c012bf5aaa3311b825d6bf92789b0707c6efd4b265d8511e20f1b2863a017934`

## 目的

在任何特征族扩展之前，只回答两个剩余问题：

1. D5/D10/D20/D40 `mfe_H` 中哪些期限提供不可被其他期限替代的信息。
2. D3/D5/D10/D20 `state_H` 是否在全部 MFE 分数之后仍有独立路径信息，以及
   raw state probability 是否给时序样本外概率指标带来增量。

本研究不训练新的 LightGBM，不修改旧预测，也不选择特征、联合损失、入场分数、
退出规则、持有期、槽位、杠杆、止损或继任架构。

## 冻结输入

- 评价年份固定为 2023、2024、2025；用户明确要求继续使用近期三折，同时承认
  它们已被复用，不宣称全新留出。
- 输入只包括 24 个既有 OOS target-horizon-year task。
- 唯一消费的 126 个 `task_result/prediction/model/daily_metrics/group_manifest/
  evaluation_rows` 文件集合哈希为
  `3c21690f813da2a4b8604375ebb7338ccf5b528f42db8ee3439fc584c888f853`。
- 最大结果日固定为 `2025-12-31`；2026 禁止加载、校准、评价或决策。
- 完整 preflight 已通过，`new_booster_count=0`。

## MFE 审计

每年分别计算期限两两相关、留一期限 residual、由短至长 residual，以及
`mfe_10-mfe_5`、`mfe_20-mfe_10`、`mfe_40-mfe_20` 新增窗口机会。ridge 系数
只使用同日预测分数，不使用结果标签。

核心 head 要求三年 residual IC 和真实尾部富集都为正，且至少两年 residual IC
不低于 0.01。次级 head 要求至少两年 IC 和尾部为正，最差 IC 不低于 -0.01。
最终等级取留一期限证据与新增窗口机会证据中的较强者；D5 只使用留一期限证据。

## State 审计

每个 state expected-score 对全部四个 MFE 分数做残差化，按真实 state ordinal IC
和 high-state Top-5% lift 使用同一 core/secondary/omit 门槛。state 与最近 MFE
真实日度三分位的 NMI/ARI 作为标签语义证据，不代替预测残余检验。

概率增量另行裁定：2024 只用 2023 OOS 分数拟合诊断逻辑回归，2025 使用
2023-2024；比较 MFE-only 与 MFE+raw-state-probability 的 multiclass Brier 和
logloss。诊断模型不保存，也不是继任模型。

## 后续边界

只有未被判为 `omit` 的 heads 才进入下一份特征族增量合同。该合同仍使用用户指定
的 2023-2025 三折，并必须在运行前一次冻结特征组、比较矩阵和多重比较规则。

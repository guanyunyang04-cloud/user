# Seq100 过程复杂度清理与概念瘦身结果

日期：`2026-07-12`

Status: `completed / guarded_process_pack_cleanup / current_facade_slimmed / winner_null_preserved / no_execution_change / no_qdp_active_change`

Active artifact impact: unchanged. `daily_research/output/active_execution_strategy.json` and QDP active pointer were not modified.

## 结论

- Verdict: targeted cleanup completed; retain the historical compatibility engine, use the slim current façade, and keep execution frozen until an execution-aligned candidate passes the four-year gates.

本轮需要清理，但不适合对训练核心做大规模重写。实际采取的是三项有边界的动作：删除无引用过程包；把当前操作热路径压缩为四步 façade；把“默认模型、机会价值、可执行收益、测试集”等混在一起的旧表述拆开。历史 registry、profile alias 和 fixed-OOS 代码仍可复现，正式 v3 所绑定的训练/生成核心未被改写。

当前研究结论没有被清理动作改变：candidate-complete v3 的 baseline/hard-ST 四年矩阵仍为 `winner=null`，禁止 freeze champion，execution 与 QDP active 保持不变。

## 物理清理

删除前清单：`daily_research/brain/references/seq100_process_pack_cleanup_inventory_20260712.json`。

- 删除对象：6 个中断的 candidate-complete v1-v6 pack、2 个 streaming smoke pack、1 个未完成的旧 daily-only pack。
- 目录数：`9`。
- 已删除字节：`26,649,845,075`（`24.8196 GiB`）。
- 删除条件：对象同时满足 partial/smoke、无脑区保留引用、无注册产物间接引用、无 retention-policy 显式保护、位于允许的 sequence-pack 根目录、不是链接/重解析点。
- 删除后 GC：`safe_delete_candidate_count=0`；9 个目标目录均不存在。

GC 同时补上了三类安全语义：

1. 新 registry 对旧 retirement/registry 的间接引用会阻止删除，因而 v1→v2→v3 证据链不会再被误判为孤儿。
2. `protected_sequence_pack_paths` 与 `protected_evidence_chain_paths` 会被 GC 实际执行，而不只是写在 policy 中供人阅读。
3. `*_cleanup_inventory` 是删除审计记录，不是保留引用；它记录目标名称但不会反过来阻止受保护 GC。

## 保护边界

删除前后均复核以下文件 SHA-256，全部未变：

| 对象 | SHA-256 |
|---|---|
| QDP active pointer | `e56f72a6cba8bcf86055817f6a0ec5e7391271fb3c27b4d628c3abc62944051e` |
| candidate-complete v7 manifest | `710b0421e554c31912ef249ca0a3df8fc1b0b3ba8b595a06784cae8fa34b7bda` |
| v3 development registry | `d8d46621d79a6d3583987c719560872db7edbe3e9411f9ab50ab435a2b4d06c8` |
| v3 result ledger | `2b547a5a427faa08cd809ecb6bdc84fce29f640a21f612d42c2ed0e29d714de2` |
| v3 development selection | `3242fb6817374f5e133bd09cb751b42bb42a40df6b5a80f67d5bf709d8585083` |
| v2 retirement | `3c6c59f6a99339d715bb3a46a4d1998ad291fbcb005970fb852ce175426e5b8d` |
| v1 retirement | `ae8eda2fb2d251fbe4f98e4d61bc9804fd5dc96b651c43fc454486d5ef40934f` |

`baseline_2022_terminal_transport_recovery.json` 也已确认 JSON 可解析、只读，SHA-256 为 `842839d8493ea84ffc1ba6b0da87543922ea0590f04df1971f4bd9fabaf0f6b5`；它记录的是 stdout transport 中断后的证据恢复，不是训练失败。

## 当前操作热路径

新增当前 façade：`daily_research.path_policy.seq100_development`。它只暴露四个有副作用的步骤：

1. `register`：冻结 source、profile、2022-2025 folds 和训练合同。
2. `run`：运行/恢复全量作业；外层继续由 `memory_guard.py --min-available-gb 1.0` 保护。
3. `select`：使用冻结资格门和排序规则。
4. `freeze`：仅在 `winner != null` 时成功。

另有只读 `contract` 用于接管。旧 `seq100_research_generation` 保留为兼容引擎，继续承载历史 screen/confirmation/fixed-OOS registry；旧 `seq100_mainline.PROFILE_SPECS` 保留 profile alias 和单模型诊断。这样不破坏历史复现，也不再让旧命令成为日常入口。

## 概念瘦身

| 当前概念 | 语义 |
|---|---|
| baseline control | 冻结对照，不是 default/champion |
| hard-ST | 已否决候选，不因实现完成而保留“待验证”身份 |
| `predicted_path_opportunity_score_v2` | 预测路径上的机会评分；`path_trade_value_v2` 仅为历史字段别名 |
| realized-plan return | 按预测入场/退出计划兑现的成本后收益 |
| live PnL | 未来真实订单结果；研究指标不自动等同于它 |
| `train/development` | 2022-2025 直接参与早停、loss 和候选选择；不是独立 test |
| historical fixed OOS | 只用于旧证据复现；若运行不得参与 checkpoint 选择 |

指标拆成三层：

- checkpoint：只看 `development_total_loss`。
- candidate eligibility：Top1/3/5/10 成本后 realized-plan alpha、stress、覆盖率、正收益年份。
- diagnosis：opportunity alpha、oracle regret、rank IC、path MAE、fill rate。

## 当前证据与下一步

v3 四年成本后 alpha 仍为：baseline Top1/3/5/10 `-2.89%/-1.83%/-1.24%/-0.06%`，hard-ST `-10.36%/-7.99%/-5.26%/-2.28%`。baseline Top3 opportunity alpha `+39.87%`，但 realized-plan alpha `-1.83%`、oracle regret `0.5651`。因此当前阻塞点是预测退出/执行目标错配，不是继续堆叠 rank 复杂度。

下一项单一研究改动是：先在同一候选和成本合同上审计固定退出、当前预测退出与 oracle executable exit；随后只增加一个 execution-aligned soft-exit 候选。该候选通过前不运行 `global_tail_512`，也不改变 active execution。

## 验证

- GC focused tests：`21 passed`。
- Current façade + development/mainline compatibility tests：`34 passed`。
- GC/store/current façade/generation/mainline/brain targeted regression：`142 passed`。
- Brain semantic guard focused tests：通过；实际 `brain_sync_audit` 为 `status=ok / finding_count=0`。
- 删除后 GC：`safe_delete_candidate_count=0`，保护哈希全部一致。
- Candidate-complete v7 pack validation、research-store verify、research consistency 均为 `status=ok`；GC 的 prediction-trim/cold-component 候选也均为 0。
- Full doc guard、integrity、evidence registry query、closure-check、Python compile 与 `git diff --check` 均通过。

# Daily Research 状态中枢

快照日期：`2026-06-19`

## 当前结论
- `daily_research` 是当前正式生产研究与执行主线；active 执行物化真源仍是 `daily_research/output/active_execution_strategy.json`。
- 当前 live/default 历史事实标签：`short_expert_policy_v5b__regoff_k1_20d_ensemble_native_anchor__active`；effective profile 为 `regoff_k1_20d_ensemble_native_anchor`，执行权重语义为 `research_raw_target_weight` / `follow_research_raw_no_global_cap`。
- 2026-05-31 接管事实：`daily_research/output/` 与 `daily_research/cache/` 曾被误删；用户确认近期 brain 结论仍正确。缺失 payload 只阻塞文件级 replay、active artifact inspection、explicit run evidence lookup 和旧 `project_consistency_check.py` 语义，不自动推翻 brain-confirmed 研究结论。
- 2026-06-01 起执行端处于 `frozen_skeleton_only / awaiting_research_rebuild`：不得恢复 active、重建 production root、生成正式 trade plan、paper/live 或 broker 接线，除非后续有显式恢复/重建授权。
- 当前 path_policy / multi-horizon 新主线为 `daily_research_v2_research_reset`；旧 Stage 2.8 / short_v5b 降级为历史先验和参考 benchmark，不再阻塞 v2 主线。
- 当前 v2 pass-grade 研究基线：`mh_v2_reset_tradeable_mainboard_anchor_20260601_01`，strict pool `policy_pool_view__925e8604a91a9c07a5387fb1`，dataset `policy_input_bundle__45e3d8c059ba718426a9f887`，feature profile `raw_kline_context_v2_tradeable_amount_checked`；该基线不是 promotion-grade，执行端仍冻结。
- stricter traditional-PIT comparison anchor `mh_v2_traditional_pit_tradeable_mainboard_anchor_20260602_01` 为 `near_pass`，不替换当前 v2 strict pass anchor。
- v2 score bridge、candidate review、bad-month attribution、risk overlay、selective throttle、state sizing、local-state input/loss、horizon concentration repair 和 horizon train-contract 已打通为 research-only 证据链；`horizon_30d_soft_penalty_v1` 已三 seed forecast gate pass，并完成 research-only score bridge / candidate matrix / bad-month attribution / selective throttle review。结论：raw candidate matrix 为 `0/27` promotion-review eligible；high-volatility throttle 显著修复 deep bad month 与 drawdown，但仍为 `0/8` promotion-review eligible，剩余 blocker 是月度正胜率不足和收益集中，不是执行解冻依据。
- v2 architecture fusion/capacity upgrade 已完成 Tier 1/Tier 2 初审：Tier 1 `hybrid_expert_fusion_static_context` 三 seed forecast pass 但弱于 GRU-static 主线，禁止 bridge；Tier 2 `regime_routed_multi_expert_horizon_v1` 已实现并完成 formal seed7 diagnostic，主评分 rank/spread 弱于 GRU-static seed7 且资源成本高，seed11/19 已暂停。当前 blocker 是 `score_mapping_and_router_calibration`，不是执行解冻依据。
- Traditional Baostock v2.1 长样本数据扩充已 research-only 导入 daily_research lake：dataset `policy_input_bundle__2082fee5bb1760972d8c9012`，same-period pool `policy_pool_view__d7a56d5164470b590e4f5a40` 和 long-history pool `policy_pool_view__eb690dd0becc330f029c21bd` 均通过 v2 contract audit；augmented feature profile smoke memmap 通过，但这不是模型质量证据。
- canonical 数据集、active registry、policy bundle 和 sharded memmap 的当前事实权威已迁入 QDP；本分脑中 `policy_input_bundle__45e3d8c059ba718426a9f887`、`policy_input_bundle__2082fee5bb1760972d8c9012` 等只表示 daily_research 研究消费 lineage 或历史 explicit dataset，不代表共享数据平台 owner。
- 2026-06-16 QDP pack frontier reconciliation 已完成：8 个 2026-06-13/14 QDP-pack / QDP-sharded runs 已登记为 `style_structural_v1` consumption evidence；最新 QDP active sharded 与 training pack 已切到 `style_structural_alpha_v2_label_v2`，feature count `307`，label schema `path20_basic_v2` version `2`，static context `symbol/exchange/industry`。`qdp_alpha_v2_label_v2_pack_consumption_smoke_20260616_01` 已用 daily protocol 快路径完成 64/64/64 role split smoke；status/training `completed`，verdict `forecast_failed` 仅表示 smoke 不证明模型强。active artifact 未触碰，执行端仍冻结。
- 2026-06-16 alpha_v2 label_v2 首轮同口径强模型对照已完成：`qdp_alpha_v2_hybrid_topn_h256_t4_b512_full_e12_20260616_01` 使用 `style_structural_alpha_v2` / `path20_basic_v2` / `hybrid_expert_fusion_static_context` / h256 / `topn_excess_rank_v1` / seed7 跑满 full pack，status `completed`，verdict `forecast_test_confirmed`。训练系统 best 为 epoch2，validation/test rank_ic20 `0.076656 / 0.115449`，spread20 `0.018850 / 0.023697`；epoch4 validation rank/spread 最强但 personal_topK 稳定性较差。`personal_topk_v1` validation 选 `pred_decision_score top1 h10`，score `276.55`，validation net `2.72%`，同候选 test net `2.73%`，明显好于旧 v1 h128/h192/h256 默认候选。该结论仍是 single-seed research-only strong-model evidence，不是 multi-seed、backtest、candidate matrix 或执行解冻依据。
- 2026-06-17 用户确认后续先围绕 alpha_v2 强模型研究计划推进；计划入口为 `daily_research/brain/references/path_policy_alpha_v2_strong_model_research_plan_20260617.md`。当前决策：先强化 hybrid alpha 模型，优先做 loss/output alignment、hybrid architecture ablation、必要的真实吞吐优化；multi-seed、score-backtest bridge、candidate matrix、execution-candidate review 后置，直到出现更强 single-seed 候选或用户显式要求。
- 2026-06-18 alpha_v2 Stage 1 loss/output alignment 已完成三个 clean single-seed scouts，均未打赢 anchor。`decision_score_topk_alignment_v1` 候选 `qdp_alpha_v2_hybrid_topk_align_h256_t4_b512_seed7_20260617_02` validation `personal_topk_v1` 选 `pred_decision_score top1 h5`，score `52.37`，validation net `1.30%`，同候选 test net `1.09%`。`score_monthly_robust_v1` 候选 `qdp_alpha_v2_hybrid_monthly_robust_h256_t4_b512_seed7_20260617_02` validation 选 `pred_cum_mu_5d top1 h5`，score `11.22`，validation net `1.20%`，同候选 test net `0.72%`。`horizon_30d_soft_penalty_v1` 候选 `qdp_alpha_v2_hybrid_h30soft_h256_t4_b512_seed7_20260617_02` 预测信号最强，validation/test rank_ic20 `0.096840 / 0.123863`，spread20 `0.024875 / 0.025812`，但 validation 选 `pred_decision_score top1 h5`，score `203.00`，validation net `1.96%`，同候选 test net `0.88%`；test-only 重选 `pred_cum_mu_1d top1 h20` 可到 net `4.74%`，仅作诊断。三者都不进入 multi-seed、score-backtest bridge、candidate matrix 或 execution review；anchor 仍为 `qdp_alpha_v2_hybrid_topn_h256_t4_b512_full_e12_20260616_01`，score `276.55`、validation net `2.72%`、same-candidate test net `2.73%`。详见 `daily_research/brain/references/path_policy_alpha_v2_loss_alignment_stage1_20260617.md`。
- 2026-06-18 用户重新明确 alpha_v2 训练合同语义：训练目标必须单一清晰，checkpoint selection 应跟随训练 objective；评估可以多视角但不能替代训练合同。当前新计划为 `daily_research/brain/references/path_policy_alpha_v2_time_efficient_topk_training_contract_plan_20260618.md`：先审计旧模型按 validation_loss 重判是否有被低估 epoch，但不直接替换 anchor；随后新增 `personal_time_efficient_topk_v1`，用 `net_utility_h / h` 修复长 horizon 累计收益天然占优问题，并让新合同的 best checkpoint 按该合同 validation loss 选择。架构 ablation 后置到新 objective 干净之后。
- 2026-06-18 alpha_v2 time-efficient topK 训练合同代码已落地，详见 `daily_research/brain/references/path_policy_alpha_v2_time_efficient_topk_training_contract_implementation_20260618.md`。`forecast_training.py` 已新增 `personal_time_efficient_topk_v1` loss profile、`validation_loss` selection profile、`future_score=max_h((cum_excess_h - cost - risk_penalty_h)/h)` target helper、time-efficient prediction columns/metrics，并修复 epoch validation frame 未传 `loss_profile` 的旧语义错位。Focused tests 与 forecast_training 全文件测试通过；这只是 code-contract enablement，不是 full alpha_v2 训练结果、multi-seed、bridge、candidate matrix 或 execution evidence。active artifact 未触碰，执行端仍冻结。
- 2026-06-18 alpha_v2 time-efficient topK 合同真实 pack smoke 已完成：`qdp_alpha_v2_time_eff_topk_contract_smoke_20260618_01` 使用 QDP `style_structural_alpha_v2_label_v2` training pack、`personal_time_efficient_topk_v1`、`selection_profile=validation_loss`、64/64/64 role split、CPU 1 epoch 跑通。训练 summary `completed`，time-efficient prediction columns 与 metrics 可用，selection rule 为 `lowest_validation_loss_for_training_loss_profile_then_seed_score`。该 smoke 只证明链路可用，不证明模型强。详见 `daily_research/brain/references/path_policy_alpha_v2_time_eff_topk_contract_smoke_20260618.md`。
- 2026-06-18 alpha_v2 Stage 0 old-contract validation-loss audit 已完成，详见 `daily_research/brain/references/path_policy_alpha_v2_old_contract_validation_loss_audit_20260618.md`。结论：旧 run 的 old mixed validation loss、old selection score、rank/spread diagnostics 和 personal_topK 视角确实会选择不同 epoch，证明用户指出的“训练目标/选择/评估语义必须分离”是合理的；但没有发现足以阻塞新合同 full run 的旧 checkpoint 问题。anchor `qdp_alpha_v2_hybrid_topn_h256_t4_b512_full_e12_20260616_01` 仍作为旧合同比较基线，不升级、不桥接、不进 candidate matrix、不触碰 active/default。
- 2026-06-19 alpha_v2 time-efficient topK 首个 h256 seed7 full run 已完成：`qdp_alpha_v2_hybrid_time_eff_topk_h256_t4_b512_seed7_20260618_01` 使用 `personal_time_efficient_topk_v1` + `selection_profile=validation_loss` 跑满 full pack，best epoch `1`，best validation loss `4.415852`；epoch2-4 train loss 继续下降但 validation loss 恶化，说明 validation-loss 选择正确阻断过拟合。validation/test rank_ic20 `0.069275 / 0.106003`，time_eff_score_rank_ic `0.074451 / 0.060200`。`personal_topk_v1` validation 选 `pred_decision_score top3 h3`，validation net `0.60%`，同候选 test net `0.57%`；test-only 重选 `pred_cum_mu_5d top1 h20` net `4.23%` 仅作诊断。该 run 未打赢旧合同 anchor 的 validation-selected same-candidate evidence；anchor 仍为 `qdp_alpha_v2_hybrid_topn_h256_t4_b512_full_e12_20260616_01`。详见 `daily_research/brain/references/path_policy_alpha_v2_time_eff_topk_h256_seed7_full_result_20260619.md`。
- 2026-06-20 alpha_v2 no-symbol + prediction-first alpha-score full run 已完成：`qdp_alpha_v2_hybrid_alpha_score_no_symbol_h256_t4_b512_seed7_20260619_01` 使用 `hybrid_alpha_score_v1`、`forecast_path_v1`、`selection_profile=validation_loss`、training static fields `exchange,industry` 跑满 12 epoch full pack。best epoch `2`，best validation loss `3.413702`；train loss 从 `2.598986` 持续降到 `2.167131`，但 validation loss 从 epoch2 后整体恶化到 epoch12 `3.517288`，train-to-plateau 未出现 late validation recovery。validation/test rank_ic20 `0.112704 / 0.116119`，spread20 `0.031486 / 0.025071`；unseen-stock test rank_ic20 `0.110278` 接近 seen-stock `0.115920`，说明去掉 `symbol_id` 没有破坏泛化。`forecast_path_v1` 不输出 `pred_decision_score`，personal topK 诊断需用 alpha prediction columns；validation 选 `pred_aux_cum_20d top5 h1`，validation net `0.15%`，同候选 test net `0.18%`；test-only 重选 `pred_aux_cum_5d top3 h20` net `4.27%` 仅作诊断。结论：`no_symbol + hybrid_alpha_score_v1` 是值得继续诊断的预测器方向，但仍未替换旧合同 anchor，不进入 multi-seed、bridge、candidate matrix 或 execution review。详见 `daily_research/brain/references/path_policy_alpha_v2_hybrid_alpha_score_no_symbol_h256_seed7_full_result_20260620.md`。
- v2 high-return model discovery 已完成 long-history augmented / 主板宽池 GRU seed7 datecap2 scout、same-period control、首轮 objective/loss datecap2 scout、`topn_excess_rank_v1` 正常样本 long-history/same-period/matrix、以及 finalist seeds `7,11,19` 和 all-seed ensemble bridge。datecap2 已降级为 `thin_sample_smoke_only`：只证明链路和报告可用，不作为模型方向淘汰依据。正常样本 `topn_excess_rank_v1` 当前为最佳 high-return 候选：long-history finalist `3/3` seed 正 transfer，seed7 small-capital matrix `15/27` research-grade，最佳变体 `h30_mw080_rb10d_all_c10_5_10_regime_off` 超额年化 `0.4284`、超额 Sharpe `1.7463`、月胜率 `0.7273`、最差月 `-0.0776`。但 all-seed mean ensemble 超额年化 `0.1177`、超额 Sharpe `0.4029`，弱于单 seed median；所以当前结论是 `research_grade_candidate_partial / seed_aggregation_blocker`，仍禁止 promotion、paper/live、broker 或 active change。
- 2026-06-03 frontier reconciliation 已确认两个未登记 seed7 run 为 research / shadow-only forecast diagnostics，详见 `daily_research/brain/references/daily_research_current_frontier_compaction_20260603.md`。
- continuous_policy 当前仍是 research / shadow-only；r61-r74 已解除若干 translation / oracle / lake collapse blocker，但 training evidence、feature contract health、cash timing、source quality 与 receiver-source spread 仍未闭合。

## 当前接管入口
- 默认读取顺序：`identity_layer.md -> state_center.md -> knowledge_center.md -> continuous_policy_design_contract.md -> operations_center.md -> governance_layer.md`。
- 首选工具入口：`C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow capsule --task "<task>" --json`。
- 当前 frontier freshness：`C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow current-frontier --json`。
- 证据查询：`C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow query --q <tag|dataset_id|r_id> --json`。
- 所有 `daily_research` 程序必须显式使用 `C:/Users/ASUS/miniconda3/envs/yolos/python.exe`。
- `latest_*` 不得直接当真源；若 latest study/protocol/audit/ledger 不同源，必须使用 explicit run tag / protocol tag / dataset id。
- PowerShell 中文显示异常时，先用显式 UTF-8 复读；不得直接判定文档损坏。

## 当前主问题
- P0：冻结 live/default/promotion/active artifact；所有新线先保持 research / shadow-only。
- P1：执行端只保留骨架、只读诊断和候选评估能力；数据缺口严格阻断，不得回退旧交易日伪装“今日计划”。
- P2：保持脑区控制面简洁；长历史、完整复盘、长命令进入 `references/`。
- P3：v2 下一步是围绕 `horizon_30d_soft_penalty_v1` 的 post-throttle 剩余负月做 month-state score calibration、validation-selected throttle thresholds、sector/liquidity/volatility/horizon cap diagnostics；只有新证据证明 target-weight concentration 重新成为主 blocker 时才跑 local risk cap。
- P3a：模型容量路线下一步只能 research-only 做 Tier 2 router/score mapping 诊断或缩小/校准版 Tier 2；当前 Tier 1/Tier 2 均不得 bridge、promotion 或改 active artifact。
- P3b：数据扩充和高收益发现路线当前优先解决 `topn_excess_rank_v1` 的 seed aggregation blocker。finalist seeds `7,11,19` 已完成，`3/3` 正 excess return / excess Sharpe，但 seed7 强、seed19 正而不达 research-grade、seed11 弱；all-seed mean ensemble 没有优于单 seed median。下一步先做 validation-selected seed weighting、rank aggregation、topN vote aggregation 和 same-period multi-seed 稳定性诊断；不允许事后剔除 seed11 当作正式规则。只有 seed 聚合/稳定性重新成立后，才进入执行适配研究。smoke、pilot、datecap2、single-seed、timeout、弱 ensemble 不得写成 promotion 或 live evidence。
- P3c：QDP alpha_v2 强模型路线当前从“执行化 loss”转向“hybrid 作为预测器 / alpha-score generator”诊断。`personal_time_efficient_topk_v1` 首个 full run 和 `hybrid_alpha_score_v1 + no_symbol` full run 均已完成；后者预测 rank/spread 更强且去掉 `symbol_id` 未损害 unseen-stock 泛化，但 personal topK validation-selected same-candidate 仍弱于旧合同 anchor。当前 anchor 仍为 `qdp_alpha_v2_hybrid_topn_h256_t4_b512_full_e12_20260616_01`，不进入 multi-seed、bridge、candidate matrix 或 execution review。
- P3c-now：下一步优先诊断 alpha-score 外部 selection/scorer，而不是继续随机长跑。重点看为什么 `forecast_path_v1` 的 test-only `pred_aux_cum_5d top3 h20` 很强、但 validation 选中的是弱得多的 `pred_aux_cum_20d top5 h1`；同时考虑 rolling/multi-year validation selection 或调整 personal topK selection profile。当前仍禁止 multi-seed、bridge、candidate matrix、execution-candidate 或 active artifact 变更。
- P4：之后再推进 per-symbol reversal/volatility bucket sizing、PIT/status 合同二阶段硬化、missing/fill 语义、limit/industry/valuation optional domains。
- P5：continuous_policy 围绕 r71/r74 继续验证 receiver/deploy 平衡、cash timing、drawdown/reversal、source quality、feature contract health 与 sufficient training evidence。

## 当前优先级
- `formal`、`recent`、`promotion`、`live` 不得混写。
- smoke、dry-run、short-window check、interrupted wrapper、insufficient evidence、failed trial、realtime tail label 都不能升级为正式 verdict。
- safe screening 或代码合同结果不得改 `daily_research/output/active_execution_strategy.json`。
- failed / timeout trial 可写诊断，不得写 completed evidence；外层等待窗口耗尽本身也不得写成 failed evidence。
- outer study 未写 `study_summary.json` 时，只能写 protocol-level evidence，不能写 completed study verdict。

## 当前边界
- `daily_research/output/active_execution_strategy.json` 缺失时，守卫应给出可读阻塞，不得 traceback，也不得手工从 brain 文本重造 active artifact。
- 本地 `output/cache` 空目录骨架不等于真实 payload 恢复；guard 通过不等于旧 run 可 file-backed replay。
- cap80、旧 replay 差异、corrected near-pass、单 seed forecast pass 或单一 candidate backtest 只能作为诊断线索，不能替代 v2 gate evidence。
- 轮询 / 异步任务必须绑定足够的可观察 handle，如 PID / job id / run id、stdout/stderr、progress、summary、checkpoint、artifact、端口或 API status；观察窗口耗尽只表示本轮观测结束，进程或产物仍推进时由 agent 调整节奏继续轮询。

## 当前风险
- 文档继续堆 dated log 会削弱接管效率；当前热路径应只保留结论、边界、入口和证据索引。
- 只增加 epoch、loss 或模型宽度可能掩盖 target construction 与 allocation semantics 断点。
- 若低预算、单 seed、best epoch 贴边、缺 learning curve 或缺多 seed聚合，不得写成模型质量结论。
- 只看 target-sum closure 会掩盖 release/source/receiver dead 与 intent translation conflict。

## 最新证据索引
- 当前前沿压缩与未登记 run reconciled：`daily_research/brain/references/daily_research_current_frontier_compaction_20260603.md`。
- v2 research reset 与框架合同：`daily_research/brain/references/daily_research_v2_research_reset_20260601.md`、`daily_research/brain/references/daily_research_v2_research_framework_contract_20260602.md`。
- v2 数据集/score/candidate/risk 证据：`daily_research/brain/references/daily_research_v2_dataset_contract_upgrade_20260602.md`、`daily_research/brain/references/daily_research_v2_score_backtest_bridge_20260602.md`、`daily_research/brain/references/daily_research_v2_candidate_review_matrix_20260602.md`、`daily_research/brain/references/daily_research_v2_selective_throttle_matrix_20260602.md`。
- v2 latest model-side references：`daily_research/brain/references/daily_research_v2_local_state_input_scout_20260602.md`、`daily_research/brain/references/daily_research_v2_local_state_loss_calibration_scout_20260603.md`、`daily_research/brain/references/daily_research_v2_horizon_concentration_repair_scout_20260603.md`、`daily_research/brain/references/daily_research_v2_horizon_concentration_train_contract_scout_20260603.md`。
- v2 horizon 30d soft-penalty execution-candidate review：`daily_research/brain/references/daily_research_v2_horizon_30d_soft_penalty_execution_candidate_review_20260603.md`。
- v2 architecture fusion/capacity upgrade：`daily_research/brain/references/daily_research_v2_architecture_fusion_capacity_upgrade_20260603.md`。
- Traditional Baostock v2.1 long-sample data expansion：`daily_research/brain/references/daily_research_v2_traditional_baostock_v2_1_data_expansion_20260604.md`。
- V2 high-return model discovery v1：`daily_research/brain/references/daily_research_v2_high_return_model_discovery_v1_20260604.md`。
- V2 high-return augmented full-pool pilot：`daily_research/brain/references/daily_research_v2_high_return_augmented_pilot_20260605.md`。
- V2 high-return augmented same-period datecap2 scout：`daily_research/brain/references/daily_research_v2_high_return_augmented_datecap2_scout_20260605.md`。
- V2 high-return long-history multi-split scout：`daily_research/brain/references/daily_research_v2_high_return_long_history_scout_20260605.md`。
- V2 high-return objective/loss scout：`daily_research/brain/references/daily_research_v2_high_return_objective_loss_scout_20260605.md`。
- V2 high-return normal-sample trial plan：`daily_research/brain/references/daily_research_v2_high_return_normal_sample_trial_plan_20260606.md`。
- V2 high-return TopN normal-sample long-history result：`daily_research/brain/references/daily_research_v2_high_return_topn_normal_long_history_result_20260606.md`。
- V2 high-return TopN normal-sample same-period and matrix result：`daily_research/brain/references/daily_research_v2_high_return_topn_normal_same_period_matrix_result_20260606.md`。
- V2 high-return TopN finalist three-seed and ensemble bridge：`daily_research/brain/references/daily_research_v2_high_return_topn_finalist_three_seed_20260606.md`。
- QDP pack frontier reconciliation and alpha_v2 consumption smoke：`daily_research/brain/references/path_policy_qdp_pack_frontier_reconciliation_20260616.md`。
- QDP alpha_v2 h256 same-mouth strong-model comparison：`daily_research/brain/references/path_policy_qdp_alpha_v2_h256_comparison_20260616.md`。
- QDP alpha_v2 strong-model research plan：`daily_research/brain/references/path_policy_alpha_v2_strong_model_research_plan_20260617.md`。
- QDP alpha_v2 Stage 1 loss/output alignment running control：`daily_research/brain/references/path_policy_alpha_v2_loss_alignment_stage1_20260617.md`。
- QDP alpha_v2 time-efficient topK training contract plan：`daily_research/brain/references/path_policy_alpha_v2_time_efficient_topk_training_contract_plan_20260618.md`。
- QDP alpha_v2 time-efficient topK training contract implementation：`daily_research/brain/references/path_policy_alpha_v2_time_efficient_topk_training_contract_implementation_20260618.md`。
- QDP alpha_v2 time-efficient topK real-pack smoke：`daily_research/brain/references/path_policy_alpha_v2_time_eff_topk_contract_smoke_20260618.md`。
- QDP alpha_v2 old-contract validation-loss audit：`daily_research/brain/references/path_policy_alpha_v2_old_contract_validation_loss_audit_20260618.md`。
- QDP alpha_v2 time-efficient topK h256 seed7 full result：`daily_research/brain/references/path_policy_alpha_v2_time_eff_topk_h256_seed7_full_result_20260619.md`。
- QDP alpha_v2 no-symbol hybrid alpha-score h256 seed7 full result：`daily_research/brain/references/path_policy_alpha_v2_hybrid_alpha_score_no_symbol_h256_seed7_full_result_20260620.md`。
- continuous_policy r61-r74 证据：见 `daily_research/brain/references/r61_release_first_decision_core_v4_status_20260514.md`、`daily_research/brain/references/r65_portfolio_set_v5_status_20260514.md` 到 `daily_research/brain/references/r74_lake_behavior_quality_status_20260515.md`。
- output/cache 误删恢复边界：`daily_research/brain/references/data_lake_output_cache_loss_recovery_boundary_20260531.md`。
- 机器索引：`daily_research/brain/references/evidence_registry.json`。
## 历史归档入口
- 历史归档：`daily_research/brain/references/state_center_archive_20260510.md`、`daily_research/brain/references/state_center_history_raw_20260424.md`、`daily_research/brain/references/state_center_evidence_index_20260424.md`。

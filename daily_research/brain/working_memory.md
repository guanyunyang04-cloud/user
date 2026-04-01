# Daily Research 研究总纲

## 1. 当前默认决策
当前默认执行主线继续保持为：
- `advanced_ml_current_code_live_anchor (ma50 baseline, lgbm520 v250) + liquid500 + next_open`
- `2026-03-31` 已补完 mainboard-only execution formal revalidation：
  - live-anchor bridge `v250 @ 504 / 21 / 520` on `20210101 -> 20260327`：`full_annual_return = 18.09%`，`full_excess_annual_return = 20.83%`，`full_excess_sharpe = 1.034`，但 `weak_window_20250905_20260319_excess_sharpe = -0.116`
  - same-protocol shortlist on `20190101 -> 20260327`：static defense `expanded_v24 + v250 = 3.44% / 5.84% / 0.279`，static offense `legacy_v7 + v255 = -0.34% / 1.97% / 0.096`，best same-profile dynamic `= 3.59% / 5.99% / 0.286`，best cross-profile dynamic `= 2.23% / 4.61% / 0.222`
  - 这意味着当前 execution wrapper frontier 与用户的 `100%` 年化目标仍相距很远；默认下一步不再是继续深拧当前 wrapper，而是把研究侧更强机会集迁到 execution 候选
- `2026-03-31` 已把第一条“研究侧机会集 -> execution candidate”桥接路径接通：`daily_research/execution/run_research_candidate_trade_plan.py` 现在可以直接消费 `deep_alpha` 的 `latest_scores.csv`，并独立写出到 `daily_research/execution/output/research_candidates/`；技术 smoke `dynamic_graph_candidate_bridge_20260331_smoke` 已通过，没有覆盖生产默认 `latest_trade_plan.txt`
- 同日晚间又把“整段研究分数 -> execution formal backtest”这条桥接补通：`deep_alpha/run_deep_alpha_research.py` 现在会导出 `daily_score_panel.csv / daily_target_weight_panel.csv`，`daily_research/execution/run_research_candidate_backtest.py` 可以直接按当前 execution 口径回测它们。第一条端到端 smoke 已跑通：
  - 研究端短窗 `deep_alpha_liquid500_dynamic_graph_bridge_20260331_smoke` 自身 holdout 年化 `154.45%`
  - 但同一分数面板翻译进 execution 统一口径后，`deep_alpha_liquid500_dynamic_graph_execbridge_20260331_smoke` 只剩 `24.82%` 年化、`3.80%` 超额年化、`0.193` 超额 Sharpe
  - 这说明当前最该优先攻击的不是“有没有高收益研究信号”，而是“研究信号迁到 execution 时的翻译损耗”
  - 同一晚还做了最小翻译层快扫：只改 execution 组合构造、不改模型本身时，`holding_count=3 + max_weight=0.35 + no_market_regime_filter` 能把该 smoke 候选抬到 `34.54%` 年化、`11.89%` 超额年化、`0.577` 超额 Sharpe；说明 execution 翻译层确实有可挖空间，但离 `100%` 年化仍远
- `2026-04-01` 已把这条思路推进到正式口径：`deep_alpha_liquid500_dynamic_graph_bridge_20260401_formal_r1` 用完整训练配置、固定 `train_end=2025-03-17 / valid_start=2025-03-18 / valid_days=252` 重跑后，研究端自身 holdout 只有 `18.56%` 年化、`5.61%` 超额年化、`0.232` 超额 Sharpe。
- 同日晚间按前一晚 smoke winner 的 aggressive translation 直接 formal 化，`deep_alpha_liquid500_dynamic_graph_execbridge_20260401_formal_r1` 结果是 `12.54%` 年化、`-1.80%` 超额年化、`-0.073` 超额 Sharpe；说明 `holding_count=3 + max_weight=0.35 + no_market_regime_filter` 没有在正式口径守住。
- 随后对同一 formal 分数面板做了 12 格翻译层 quickscan；最佳组合改成 `holding_count=8 + max_weight=0.25 + no_market_regime_filter`，但也只有 `16.02%` 年化、`1.24%` 超额年化、`0.058` 超额 Sharpe。结论是：翻译层确实重要，但当前这条 `dynamic_graph_v1 -> static liquid500 execution bridge` 还不足以把 execution frontier 推到新台阶，更谈不上接近 `100%` 年化。
- `2026-04-01` 晚间已把 execution bridge 升级成双模式：`run_research_candidate_backtest.py` / `backtest_external_score_panel.py` 现在既能吃 `daily_score_panel.csv`，也能直接吃研究端导出的 `daily_target_weight_panel.csv`；同一入口会额外输出 `weeklyized_return / excess_weeklyized_return` 作为辅指标。
- 这轮正式比较说明“周化收益”不适合替代年化收益当主评估标准：在同一回测窗口里它只是年化收益的单调变换，更适合做读数辅助，不会改变 winner 排序。
- 同日晚间已对 `deep_alpha_liquid500_dynamic_graph_bridge_20260401_formal_r1` 的 `daily_target_weight_panel.csv` 做正式桥接：`deep_alpha_liquid500_dynamic_graph_execbridge_weightpanel_20260401_formal_r1 = 15.41%` 年化、`0.70%` 超额年化、`0.031` 超额 Sharpe、`-19.88%` 回撤。它明显好于默认 aggressive score 重建桥（`12.54% / -1.80% / -0.073`），但仍略低于同日晚间 score-panel quickscan best（`16.02% / 1.24% / 0.058`），所以当前结论更新为：`target_weight` 直连桥值得作为默认优先桥接表达，但 execution frontier 仍远未接近 `100%` 年化。
- `2026-04-01` 深夜又把这条桥继续打到候选交易计划：`run_research_candidate_trade_plan.py` / `generate_daily_trade_plan.py` 现在也支持 `--external-target-weight-csv`，可直接消费研究端 `daily_target_weight_panel.csv`，不再强依赖 `latest_scores.csv` 这类“先分数再重建权重”的旧表达。
- 同一轮 smoke `dynamic_graph_target_weight_candidate_20260401_smoke` 已跑通，产物写到 `daily_research/execution/output/research_candidates/`；当前 `2026-03-31` 这份研究权重最后一天本身就是全零，所以候选计划正确输出为“无明确调仓动作”。另外由于同 run 的 `daily_score_panel.csv` 只到 `2026-03-02`，计划入口已做自动回退：若外部分数上下文缺失，就用 `target_weight` 自身作为显示代理，不让显示层反向卡死直连桥。
- `2026-04-01` 深夜已把 `offset-ensemble / phase-robust execution bridge` 原生化：`backtest_external_score_panel.py`、`run_research_candidate_backtest.py`、`generate_daily_trade_plan.py`、`run_research_candidate_trade_plan.py` 现在都支持 `rebalance_offset_mode=all` 与 `rebalance_anchor_date`。这轮验证还钉死了一个新边界：如果不给固定 anchor，all-offset ensemble 的结果会受“历史起点从哪天开始加载”影响，不能直接当生产逻辑。
- anchored native formal finalists 已在同一 bridge window 下复核完成，当前最强两条 execution candidate 是：
  - `deep_alpha_liquid500_dynamic_graph_execbridge_regon_k1_10d_ensemble_native_anchor_20260401_formal_r3 = 52.32%` 年化、`32.91%` 超额年化、`1.800` 超额 Sharpe、`-9.36%` 回撤
  - `deep_alpha_liquid500_dynamic_graph_execbridge_regoff_k2_10d_ensemble_native_anchor_20260401_formal_r3 = 52.14%` 年化、`32.75%` 超额年化、`2.188` 超额 Sharpe、`-7.72%` 回撤
- `regoff_k2_10d` 与 `regon_k1_10d` 的 manual ensemble verify 也已补齐，说明 native anchored bridge 与旧的手工 ensemble 结果已经基本贴合；因此当前默认生产候选应升级为“anchored offset-ensemble”，而不是继续回头拧 `score -> weight` 翻译器。
- `2026-04-01` 上午已补完 exact same-window live-anchor replay：`advanced_ml_live_anchor_samewindow_20260401_formal_r1` 里的 `trend_up_low_vol_ml25_none25_v250` 在 `bridge_full (2025-03-18 -> 2026-03-31)` 上是 `24.34%` 年化、`10.75%` 超额年化、`0.580` 超额 Sharpe、`-14.42%` 超额回撤；因此 anchored ensemble 对当前 live-anchor 的优势现在已经不是“近似窗口推断”，而是同窗正式结论。
- `2026-04-01` 上午已把第一版 soft state-conditioned sizing 直接接进 `target_weight` 直连桥和 candidate trade plan，并在 `execution_target_weight_state_conditioned_verdict_20260401_r1` 里把两个 finalists 跑完：
  - `regoff_k2_stateoff` 仍是默认 winner；`quadrant_guard_v1 / trend_guard_v1 / market_state_guard_v1` 都会把年化从 `52.14%` 压到 `45.56%~46.72%`，虽然回撤能收窄到 `-6.28% ~ -6.82%`，但不够抵消收益损失。
  - `regon_k1_stateoff` 仍是更激进的收益对照；`quadrant_guard_v1` 在当前 hard regime filter 下基本是 no-op，`trend_guard_v1 / market_state_guard_v1` 会把年化从 `52.32%` 压到 `50.47%`，只换来约 `1.5pct` 的回撤改善。
  - 因此当前结论更新为：soft state-conditioned sizing 有风险塑形价值，但还没有拿到默认升级资格；默认生产候选继续保持 `regoff_k2_10d_ensemble_native_anchor`，`regon_k1_10d_ensemble_native_anchor` 保留为收益上沿对照。
- `2026-04-01` 深夜继续把“翻译损耗”方向一次性扫到底后，当前结论已经很明确：
  - 仅在 `1d` 直连权重桥里做 `power / min_weight / top_k=4/5 / full_invest` 这类小修小补，收益几乎不动；`15.41% / 0.70% / 0.031` 只会在小数点附近摆动。
  - 真正的大头来自“执行节奏 + 直连权重表达”：`top_k=2` 与 `top_k=1` 集中化在 `1d` 已能把桥接 formal 提到 `21.30% / 5.84% / 0.219` 或 `19.54% / 4.31% / 0.125`。
  - 进一步把 `rebalance_freq` 扩到 `5d / 10d` 后，单 offset 点估值可以冲到 `97%~117%` 年化，但相位敏感性非常大；不同 offset 下最差会掉回负超额，因此“单一幸运 offset”不具备生产资格。
  - 但当把所有 offset 做成等权 sleeve ensemble 后，收益依然很强且明显更可用：`regon_k1_10d ensemble = 51.80%` 年化、`32.45%` 超额年化、`1.779` 超额 Sharpe、`-9.28%` 回撤；`regoff_k2_10d ensemble = 50.46% / 31.28% / 2.105 / -7.72%`；即使不做集中化，`regoff_raw_10d ensemble` 也有 `40.05% / 22.20% / 1.680`。
  - 因此这条方向的最终 verdict 不是“没用”，而是“方向正确且有明显应用价值，但要从 `daily target_weight + slower/staggered execution cadence` 里拿价值，不能把单 offset lucky run 当生产答案”。默认下一步应转向 `offset-ensemble / phase-robust execution bridge`，而不是继续回去拧旧的 `score -> weight` 翻译器。
但 `2026-03-29` 的代码考古 + 受控 ablation 已钉死一个关键红旗：
- 在隔离 worktree `H:/new_tdx64/PYPlugins/user_ablation_labelgap_off` 中，只把 `daily_research/baseline/ml_alpha.py::_label_lookahead_bars()` 临时改成 `return 0`，同口径 `legacy_v7 + no_auto_trim_history + liquid500 + next_open + lgbm` 就会从当前代码的 `151.70% / 0.882` 立即回跳到旧快照的 `958.89% / 2.447`
- 当前与快照的 `features.py`、`build_ml_target()` 一致，因此旧快照高收益主因不是“因子更强”或“目标公式不同”，而是 `next_open` 训练边界未做 label-safe gap，存在严重 `label leakage / look-ahead bias`；`2026-03-29` 已把执行 wrapper 从旧快照后端切回当前仓当前代码执行链路，并在同口径 bridge validation 后升级到 `lgbm520` live anchor，不再继续默认沿用那条 `958.89%` artifact 链路
当前默认口径同时固定为：

- 股票池：
  - 执行端日常使用 `liquid500_latest.txt`
  - 正式研究使用历史滚动 `liquid500 / liquid800`
- 成交假设：
  - `next_open`
- 调仓语义：
  - 日频目标更新，主入口默认 `rebalance_freq=1d`
- 状态门控：
  - `regime_ma_window=50`
  - `regime_max_annual_vol=0.32`
  - `trend_up_low_vol,trend_up_high_vol`
- 模型族：
  - `lgbm`
- 默认训练窗口：
  - `ml_train_window_days=504, lgbm_n_estimators=520`
- 默认执行后端：
- 当前执行后端：
  - `daily_research/baseline/train_trade_model.py`
  - `daily_research/baseline/generate_daily_trade_plan.py`
  - 继续写回当前 `daily_research/execution/models/` 与 `daily_research/execution/output/`
- 默认状态集成：
- 当前执行 wrapper 默认注入：
  - `trend_up_low_vol=ml:0.25,none:0.25,v2:0.50`
  - `enhanced_profile=up_low_breakout_v2`

## 2. 当前最近待决策事项
用户已在 2026-03-28 明确纠偏项目主目标：

- 坏市场收益、弱窗口收益、稳定性提升，都必须建立在“不降低当前收益率”的前提下
- 不同阶段可以用不同策略
- 但每个阶段都必须追求最大利润，不能靠委曲求全把更低收益方案升成执行端

这意味着当前执行端主目标已重新收口为：

- 收益峰值优先
- 稳定性 / 坏市场 / 弱窗口修复只能作为附加增益或阶段控制器约束
- 任何单纯“更稳但更低收益”的候选，不再自动具备执行端升级资格

用户随后又在 2026-03-28 明确要求：

- “我要的就是最高收益”

因此当前执行端决策已经进一步改写为：

- 不再先守当前代码口径下的 live-defense 默认值；
- 已把执行端从已证伪的旧高收益快照后端切回当前仓当前代码后端，并完成 `v250 @ 504 / 21 / 260 -> 520` 的同口径 bridge validation；
- 当前代码口径下的 `v250 / v255 / 双 profile 控制器` 结论继续保留为研究侧比较锚点；其中 `v250 @ 504 / 21 / 520` 已是当前执行默认值，`v255` 仍是 offense comparator。

2026-03-28 的当前代码口径 formal R3 重跑，之前曾把执行端短期升级收口为一个明确结论：

- live 默认值升级为 `trend_up_low_vol_ml25_none25_v250 @ 504 / 21 / 520`
- `trend_up_low_vol_ml25_none20_v255 @ 504 / 21 / 520` 保留为进攻对照与未来动态控制器的 `offense leg`

当前明确结论：

- `daily_research/output/advanced_ml_attack_defense_controller_20260328_formal_r3_weightgrid_focus` 与 `daily_research/output/advanced_ml_attack_defense_controller_20260328_formal_r3_baseprobe` 已经产出；
- 当前静态 `v250`：`full_excess_sharpe = 0.759`，`weak_window_20250905_20260319_excess_sharpe = 1.073`，`trend_up_low_vol_weak_window_20250905_20260319_excess_sharpe = 1.083`
- 当前静态 `v255`：`full_excess_sharpe = 0.675`，`weak_window_20250905_20260319_excess_sharpe = 0.281`，`trend_up_low_vol_weak_window_20250905_20260319_excess_sharpe = 0.094`
- 当前 base-equivalent 默认 artifact：`full_excess_sharpe = 0.294`，`weak_window_20250905_20260319_excess_sharpe = 0.151`，`trend_up_low_vol_weak_window_20250905_20260319_excess_sharpe = -0.186`
- 最强动态候选最多只做到 `0.791 / 0.992 / 0.980`，仍没有同时压过新的 live 默认值与进攻对照
- 旧 formal `r1 / r2` 里“`v255` 更适合作为默认主候选”的口径已失效，因为底层 `market_features` 已从 `7` 扩到 `24`，rolling ML scores 已发生实质变化
- 下一步不再回头争论旧 split verdict，也不再重复 `504 / 5 / 260` 与 `378 / 21 / 520` 的正式复验；
- 后续正式方向收口为：
  - 把动态控制器的 benchmark 改成 live `v250` 默认值；
  - 用 `v255` 继续承担进攻对照。

两组候选的当前职责已经重新收口为：

- 当前 live default：
  - `trend_up_low_vol_ml25_none25_v250 @ 504 / 21 / 520`
- 当前 offense comparator：
  - `trend_up_low_vol_ml25_none20_v255 @ 504 / 21 / 520`
- 当前 upgrade gate：
  - 只有动态方案在同一 formal 口径下跑赢 live `v250`，同时不丢失 `v255` 的 full 端进攻价值，才允许继续讨论默认值升级
- `market_feature_profile_compare_20260328_formal_r1` 已进一步确认：
  - `expanded_v24` 对 `base_global` 是明显修复；
  - 对 live `v250` 是“全样本几乎持平、弱窗口更强、full Sharpe 略低”；
  - 对 `v255` 与当前动态控制器则明显更弱。
- `market_feature_profile_pruning_20260328_formal_r1` 已继续确认：
  - 简单的 `market_features` 裁剪没有产生新的单一升级赢家；
  - `continuous_quadrant_v9` 虽然能把 `v255` 的弱窗口拉回一部分，但会明显牺牲 full 端，并拖累 live `v250`；
  - `legacy_v7` 仍是当前最强 offense profile，`expanded_v24` 仍是当前最强 live-defense profile。
- `market_feature_stack_ab_20260328_formal_r1` 已把“旧 offense 栈 vs 当前 live 栈”跑清：
  - 旧 `legacy_v7 + v255`：`full_annual_return = 15.54%`，`full_excess_annual_return = 18.22%`，`full_excess_sharpe = 0.860`
  - 当前 `expanded_v24 + v250`：`13.92% / 16.57% / 0.759`
  - 当前 live 相对旧 offense：`full_excess_sharpe -0.100`，但 `recent_full_excess_sharpe +0.030`，`weak_window_20250905_20260319_excess_sharpe +0.254`，`trend_up_low_vol_weak_window_20250905_20260319_excess_sharpe +0.332`
  - 所以这次不是“整体升级”，也不是“白改了全面变弱”，而是把 full 端进攻上沿换成了更强的 live 稳健性。
- `advanced_ml_model_family_compare_20260319_legacy_snapshot_reaudit_r2` 已把旧高收益日志做成可审计证据：
  - 通过 `git` 历史快照 `e7d0f8d (2026-03-24 18:58:23 +0800)` 直接重跑旧脚本；
  - `lgbm` 复刻结果为 `full_excess_total_return = 910.30%`，`full_excess_sharpe = 2.398`
  - 这与旧日志里的 `887.75% / 2.300` 已属于同一量级，说明旧高收益在旧系统里是真实结果，而不是凭空写出来的假数字。
- 但 `2026-03-29` 的受控 ablation 又进一步说明：
  - “旧系统里能稳定复刻出 9x 收益”这件事本身是真实的；
  - 但它的主因是 `next_open` 训练边界缺少 label-safe gap，而不是可直接继承到今天执行端的真实 alpha。
- 但在用户刚刚明确的“收益优先非降级”目标下，`expanded_v24 + v250 @ 504 / 21 / 520` 只应被表述为当前执行默认值，而不是项目探索的终局停止点；
  它更准确的身份是：
  - 当前代码口径下的防守型 live 方案
  - 当前默认值，但仍允许在“不降级”的前提下继续被更优方案替换
- `2026-03-29` 已完成 `advanced_ml_cross_profile_attack_defense_20260329_formal_r1`：
  - 静态 defense `expanded_v24 + v250`：`full_annual_return = 13.92%`，`full_excess_sharpe = 0.759`，`weak_excess_sharpe = 1.073`，`focus_weak_excess_sharpe = 1.083`
  - 静态 offense `legacy_v7 + v255`：`full_annual_return = 15.54%`，`full_excess_sharpe = 0.860`，`weak_excess_sharpe = 0.819`，`focus_weak_excess_sharpe = 0.751`
  - 最强 cross-profile dynamic：`full_annual_return = 15.49%`，`full_excess_sharpe = 0.836`，`weak_excess_sharpe = 1.359`，`focus_weak_excess_sharpe = 1.458`
- 这次 formal 结果说明：
  - 双 profile 控制器确实把弱窗口与 focus-weak 防守补强了；
  - 但它没有把 full 年化推过静态 offense 的 `15.54%`，也没有形成同时压过两组静态控制的单一赢家；
  - 因此按用户的“若控制器不能把年化抬出新台阶，就转向新机会集”规则，当前 `v250 / v255 / 双 profile 控制器` 参数空间不再是第一研发前线。
- `2026-03-31` 的 mainboard-only revalidation 已进一步确认：
  - 上面这组 `13.92% / 15.54% / 15.49%` 现在只能保留为旧 universe comparator，不再代表当前 execution frontier
  - 在新 universe 下，同口径 `20190101 -> 20260327` 已退化为：static defense `3.44% / 0.279`，static offense `-0.34% / 0.096`，best same-profile dynamic `3.59% / 0.286`，best cross-profile dynamic `2.23% / 0.222`
  - 所有 `weak_window_20250905_20260319` 与 `trend_up_low_vol_weak_window_20250905_20260319` Sharpe 都转负，因此当前 execution shortlist 不再接近“收益优先非降级”的升级条件

## 3. 当前项目判断
- `advanced_ml` 继续承担当前正式执行职责，执行后端已稳定落在当前仓当前代码 live-anchor 口径。
- `deep_alpha` 仍是长期主研究线，而当前最近待决策事项已经从“执行端升级 shortlist 的最终判决”转成“寻找能把 clean annual frontier 抬出新台阶的新机会集”。
- 当前 `advanced_ml` 的 `v250 / v255 / 双 profile 控制器` frontier 已基本画清：
  - 控制器可以提升稳健性；
  - 但没有把年化前沿抬过静态 offense；
  - 继续深挖这组参数的边际收益已明显下降。
- `2026-03-29` 已把“新 alpha 家族”推进到 formal entry：`daily_research/output/deep_alpha_minimal_matrix_20260329_backbone_r1` 已连续完成 `backbone / score_head / ranking`，winner 仍是 `enc-patch__pre-nopre__score-manual__rank-plain`，`selection_scope = eligible_only`，`mean_excess_sharpe = 0.744`，`mean_excess_total_return = 60.76%`
- `patch_transformer` 无预训练版本当前优于 `gru`；`masked pretrain`、`ridge`、`lgbm` head、`ranked` loss 都没有把前沿抬高，因此这一轮最小矩阵已经说明：当前不该继续在 `score_head / ranking` 细旋钮上久留
- `2026-03-29` 已补完 `relation` 正式阶段：`norel` 仍优于 `rel`，`mean_excess_sharpe = 0.744 > 0.413`，因此 `relation_layer` 在 rolling `liquid500` 下正式降级。
- `2026-03-30` 已定位并修复 `run_deep_alpha_research.py` 的 strict-window bug：旧版 `valid_days` 只约束了 `valid_start`，却没有真正截断 `valid_end`，导致之前的 `liquid800` “多窗口”结果其实是嵌套长 holdout，不是严格等长 walk-forward。
- 按修正后的 strict walk-forward 重跑 `rolling liquid800` 后，当前 clean frontier 明确收口为单一 winner：`liquid800_plain = 51.02% / 46.73% / 1.621 / 1.124`，显著强于 `liquid800_ranked = 35.50% / 29.31% / 1.173 / 0.184`；旧的“`plain` 稳定前沿 + `ranked` 激进前沿”口径失效，`ranked` 只保留为次级对照。
- 同一轮 strict `liquid800` 下，`relation_layer` 与 `masked_pretrain` 也都正式失败：`relation = 11.37% / 4.73% / 0.061 / -1.520`，`masked_pretrain = 17.67% / 12.85% / 0.477 / -0.346`。
- `2026-03-31` 已在新 universe 约束下补完 `patch_plain vs original_mamba_plain` strict head-to-head：研究范围固定为“上证 A + 深证 A，剔除创业板 / 科创板 / ST”，结果 `patch` 在 `3/3` 窗口全胜；均值上 `patch = 3.66% / 3.71% / 0.137 / -23.84%`，`original_mamba = -4.39% / -4.41% / -0.233 / -25.60%`，因此原始 `mamba` 正式记为“新 universe 下首轮失败”，不再直接进入优化循环。
- 同日已把 `dynamic graph v1` 接入 `deep_alpha` 正式脚本：新增 `--dynamic-graph-layer` 与 `top-k / temperature / industry_boost / style_boost` 参数，形态是“按日重算 top-k peer graph 特征层”，并已跑通技术 smoke `daily_research/output/deep_alpha_liquid800_dynamic_graph_20260331_smoke_v2`；该 smoke 只代表链路可用，不参与 frontier 判决。
- 随后已补完新 universe 下的 strict formal head-to-head：`relation_baseline` 对 `dynamic_graph_v1`，产物在 `daily_research/output/deep_alpha_relgraph_h2h_20260331_mainboard_r1`。结果 `dynamic_graph_v1` 以 `2/3` 窗口 Sharpe 胜、`3/3` 窗口总收益胜通过旧 `relation` baseline；均值上 `dynamic_graph_v1 = 15.48% / 15.64% / 0.534 / -25.71%`，`relation_baseline = -4.09% / -4.10% / -0.117 / -27.47%`。
- 随后又补完 `plain vs dynamic_graph_v1` 的 strict head-to-head，汇总在 `daily_research/output/deep_alpha_plain_vs_dynagraph_h2h_20260331_mainboard_r1`；该汇总复用了已存在且口径一致的 formal `plain` 与 `dynamic_graph_v1` 三窗口结果，没有重复训练。结果 `dynamic_graph_v1` 已在均值上正式跑赢当前 `plain` 主前沿：`dynamic_graph_v1 = 15.48% / 15.64% / 0.534 / -25.71%`，`plain = 3.66% / 3.71% / 0.137 / -23.84%`；按窗口看是 `2/3` Sharpe 胜、`2/3` 总收益胜。
- 因此 `dynamic_graph_v1` 当前应上升为 strict rolling `liquid800` 的 leading branch；但它的均值回撤仍深于 `plain`，且第一个窗口总收益未赢，所以默认下一步不是直接切 `MoE`，而是继续做 `dynamic graph` 的稳健性 / 参数确认。
- `plain` 与 `ranked` 的近似 stitched-return controller 也已补完：walk-forward state/window controller 没有跑赢 `plain`；只有 oracle / window-oracle 略高于 `plain`，因此这条控制线目前不再是第一优先级。
- 底层市场状态层已经完成架构升级：`quadrant` 继续保留为兼容标签；新默认研究输入已同时提供 `benchmark_trend_gap / benchmark_vol_gap / benchmark_vol_ratio / trend_bucket / vol_bucket / market_state`；后续策略默认优先接 `regime_state_selector`，而当前 `state_alpha_profile` 若切到更细 selector 会显式报错而不是静默退化。
- `base_global` 即使经过参数优化，也没有通过当前弱窗口修复门槛，不再作为执行升级主候选。
- `504 / 5 / 260` 与 `378 / 21 / 520` 已不再是优先晋级参数组合。
- `none / v2` 保留为规则层先验，不扩成新的执行主线。
- 连续状态分数保留为诊断层，不进入默认执行软调节逻辑。
- 坏市场专项若重启，主目标固定为“坏市场绝对收益”，不得再用坏市场超额替代坏市场盈利。

## 4. 当前优先级
### 优先级 A：转向新机会集
目标：
- 在执行端继续稳定保留 `advanced_ml_current_code_live_anchor (ma50 baseline, lgbm520 v250)` 的前提下，停止继续深挖当前 `v250 / v255 / 双 profile 控制器` 参数空间；
- 把研发主线转到“有机会把年化真正抬出新台阶”的新机会集或新 alpha 家族；
- 继续遵守“利润优先非降级”，不接受只补稳健、不抬前沿的延长战。
具体动作：
1. 把 `advanced_ml_cross_profile_attack_defense_20260329_formal_r1` 视为当前参数空间的 frontier map，而不是新的默认升级起点。
2. 记住这组三元边界：
   - 静态 defense `expanded_v24 + v250 = 13.92% / 0.759 / 1.073 / 1.083`
   - 静态 offense `legacy_v7 + v255 = 15.54% / 0.860 / 0.819 / 0.751`
   - 最强 dynamic `= 15.49% / 0.836 / 1.359 / 1.458`
3. 因为 best dynamic 仍未突破静态 offense 的年化 `15.54%`，当前 `v250 / v255 / controller` family 暂不继续加大扫参投入。
4. 后续“新机会集”至少应体现为以下之一：
   - 新 universe / 新容量假设
   - 新 alpha family / 新表示学习主线
   - 新组合构建与收益翻译方式
5. 所有新机会集仍必须走：
   - 历史滚动高流动性股票池
   - `next_open`
   - 多窗口 walk-forward
6. 在新机会集没有跑出明确新前沿前，执行默认值不变，不允许因为“旧 offense 更高”或“动态更稳”而反复摇摆 live anchor。
7. 当前这条“新 alpha 家族”主入口已经完成最小矩阵正式收口：`daily_research/deep_alpha/run_minimal_matrix.py --phase backbone / score_head / ranking` 都已落盘，winner 始终保持 `patch + no pretrain + manual + plain`。
8. 这条线当前最大的时间瓶颈不是矩阵包装层，而是每个 window 里的 `run_deep_alpha_research.py` 全流程重训，尤其是 `[4/8] Building sequence features and targets` 与 `[6/8] Training deep alpha model`；后续新研究必须优先考虑复用 per-window cache 提速。
9. `deep_alpha` 当前更强的新机会集仍是 strict rolling `liquid800`，但 frontier 已不再分裂成 `plain/ranked` 双主线；修正后应把 `liquid800_plain` 视为唯一主前沿，`ranked`、`controller`、`hold3_w40` 只保留为已验证但未晋级的旁支对照。

### 优先级 B：维持当前执行主线稳态
目标：
- 保持当前默认执行链路可用、可复核、可回滚。
具体动作：
1. 继续依赖 `latest_ml_model.json` 的验证摘要与模型新鲜度保护。
2. 继续把执行端默认值明确解释为：
   - `advanced_ml (ma50 baseline, lgbm) + liquid500 + next_open`
3. 若后续继续做执行端长窗口正式复核，优先沿当前 shortlist 收口，不回退到更早已淘汰参数。
4. 若重启坏市场专项，统一按“坏市场绝对收益”排序与验收。

### 优先级 C：用 `deep_alpha` 承接新机会集探索
目标：
- 把 `deep_alpha` 作为“新机会集 / 新 alpha 家族”的第一承接点之一；
- 在不打断当前执行默认值的前提下，继续推动真正可能抬升年化前沿的研究线。
具体动作：
1. 固定正式框架：
   - 历史滚动高流动性股票池
   - `next_open`
   - 多窗口 walk-forward
2. 当前 backbone 已选出 `patch_transformer + no pretrain + manual + plain`。
3. 在 strict rolling `liquid800` 这个更强新机会集里，当前唯一主前沿是 `liquid800_plain`；`ranked`、`controller`、`relation_layer`、`masked_pretrain` 与 `hold3_w40` 都已在 formal 或 strict formal 下失利。
4. `2026-03-31` 的新 universe strict head-to-head 也已把原始 `mamba` 降级为首轮失败分支；这条 backbone 先记结论，不再直接追加优化。
5. 后续若继续做 `deep_alpha`，默认顺序收口为：先围绕 `dynamic_graph_v1` 做稳健性确认与图参数 / 先验消融；若动态图主线最终没法稳定守住新前沿，再切到 `state-conditioned MoE`；不要继续围绕 `plain/ranked/controller`、旧 `relation_layer` 或原始 `mamba` 做小修小补。
6. 新分支仍应继续优先复用 per-window cache 与已落盘 artifact。

### 优先级 D：把治理规则变成日常流程
目标：
- 避免语义记忆、项目地图、工作记忆、环境模型、行动系统与情景记忆再次混写。
具体动作：

1. `semantic_memory.md` 只写当前状态、入口与边界。
2. `project_map.md` 只写项目背景、当前瓶颈与未来方向。
3. `working_memory.md` 只写当前默认决策、shortlist、优先级与停止规则。
4. `environment_model.md` 只写解释器、依赖与推荐调用口径。
5. `action_system.md` 只写执行流程与日常操作细节。
6. `episodic_memory.md` 只写时间顺序实验记录，并保持日期顺序追加。

## 5. 当前不推进或已降级的方向
- `base_global` 参数优化：
  - 可以作为对照，但不再是当前执行升级主候选。
- `504 / 5 / 260`：
  - 仍可视为全样本进攻型对照点，但不再是优先晋级参数。
- `378 / 21 / 520`：
  - 未通过长窗口正式复验，不再继续推进。
- `ma47/48` 左侧边界带：
  - 当前判断为季度特定槽位命中，不再晋级执行端。
- `v21_volume_contraction_015`：
  - 保留为规则层研究候选，但不正式晋级默认执行口径。
- 连续状态软调节：
  - 保留为诊断层，不进入当前执行层默认逻辑。
- 状态专属模型族大范围重启：
  - 在当前 shortlist 已经收敛的前提下，暂不重开更大分支。

## 6. 已明确的边界
- 当前默认执行主线不是笼统的：
  - `advanced_ml + liquid500 + next_open`
- 而是更精确的：
  - `advanced_ml (ma50 baseline, lgbm) + liquid500 + next_open`
- 任何准备接近执行端的新方案，都必须先过：
  - 历史滚动股票池
  - `next_open`
  - 多窗口 walk-forward
  - 强窗口不被破坏
- 当前执行升级 shortlist 的后续结论，继续统一使用：
  - `weak_window_20250905_20260319`

## 7. 停止规则
以下情况一律不晋级执行端：

1. 只修复最弱窗口，却破坏原本更强窗口。
2. 只在静态最新股票池上好看，未通过历史滚动股票池验证。
3. 只在单次样本或烟测中有效，未通过多窗口 walk-forward。
4. 收益改善主要来自季度集中或少数日期放大。
5. 结论无法在 `episodic_memory.md` 中回溯到对应实验与产物。
6. 所谓“坏市场盈利”若只体现为坏市场超额更高、但坏市场绝对收益仍持续为负，不得按坏市场盈利方案晋级。
7. 当前 shortlist 的 head-to-head 若不能在同一正式口径下稳定分出胜负，则维持现默认值，不做口头升级。
8. 若 cross-profile 攻守控制器已在同口径 formal 下证明“稳健性提升但年化不创新高”，则停止继续把当前参数空间当作第一研发前线。

## 8. 文档维护规则
- `daily_research/brain/semantic_memory.md`
  - 只保留当前状态、主入口与脑边界。
- `daily_research/brain/brain_architecture.md`
  - 只保留分脑结构、写入路由与父子脑关系。
- `daily_research/brain/project_map.md`
  - 只保留项目背景、当前瓶颈、未来方向与协作导航。
- `daily_research/brain/working_memory.md`
  - 只保留当前默认决策、升级 shortlist、优先级与停止规则。
- `daily_research/brain/procedural_memory.md`
  - 只保留可复用方法学与技能。
- `daily_research/brain/environment_model.md`
  - 只保留解释器、依赖与推荐调用方式。
- `daily_research/brain/action_system.md`
  - 只保留盘后执行链路与日常操作细节。
- `daily_research/brain/episodic_memory.md`
  - 只保留时间顺序实验记录与阶段结论。
- `brain/master_brain.md`
  - 负责工作区级主脑治理说明。

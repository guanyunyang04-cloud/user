# Daily Research 研究总纲

## 1. 当前默认决策
当前默认执行主线继续保持为：
- `advanced_ml_current_code_live_anchor (ma50 baseline, lgbm520 v250) + liquid500 + next_open`
但 `2026-03-29` 的代码考古 + 受控 ablation 已钉死一个关键红旗：
- 在隔离 worktree `H:/new_tdx64/PYPlugins/user_ablation_labelgap_off` 中，只把 `daily_research/baseline/ml_alpha.py::_label_lookahead_bars()` 临时改成 `return 0`，同口径 `legacy_v7 + no_auto_trim_history + liquid500 + next_open + lgbm` 就会从当前代码的 `151.70% / 0.882` 立即回跳到旧快照的 `958.89% / 2.447`
- 当前与快照的 `features.py`、`build_ml_target()` 一致，因此旧快照高收益主因不是“因子更强”或“目标公式不同”，而是 `next_open` 训练边界未做 label-safe gap，存在严重 `label leakage / look-ahead bias`；`2026-03-29` 已把执行 wrapper 从旧快照后端切回当前仓安全桥接后端，不再继续默认沿用那条 `958.89%` artifact 链路
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
- 当前安全桥接后端：
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
- 已把执行端从已证伪的旧高收益快照后端切回当前仓安全桥接后端；
- 当前代码口径下的 `v250 / v255 / 双 profile 控制器` 结论继续保留为研究侧比较锚点，而不是当前执行默认值。

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
- 但在用户刚刚明确的“收益优先非降级”目标下，`expanded_v24 + v250` 不再能被直接表述成终局正确答案；
  它更准确的身份是：
  - 当前代码口径下的防守型 live 方案
  - 不是已经满足用户原始目标的利润最优方案
- 因此后续若要同时满足用户目标，研发方向应从“继续做全局 pruning”收口到“双 profile 攻守控制器”：
  - offense leg 优先研究 `legacy_v7`
  - defense/live leg 继续保持 `expanded_v24`

## 3. 当前项目判断
- `advanced_ml` 继续承担当前正式执行职责，但执行后端已切回当前仓安全桥接口径。
- `deep_alpha` 仍是长期主研究线，但当前最近待决策事项已经切到执行端升级 shortlist 的最终判决。
- 底层市场状态层已经完成架构升级：
  - `quadrant` 继续保留为兼容标签
  - 新默认研究输入同时提供 `benchmark_trend_gap / benchmark_vol_gap / benchmark_vol_ratio / trend_bucket / vol_bucket / market_state`
  - 后续新策略与新诊断默认优先接 `regime_state_selector`，不再新增硬编码四象限分支
  - 但当前 `state_alpha_profile` 仍只支持 `quadrant`；若切到更细 selector，系统现在会显式报错而不是静默退化
- `base_global` 即使经过参数优化，也没有通过当前弱窗口修复门槛，不再作为执行升级主候选。
- `504 / 5 / 260` 与 `378 / 21 / 520` 已不再是优先晋级参数组合。
- `none / v2` 保留为规则层先验，不扩成新的执行主线。
- 连续状态分数保留为诊断层，不进入默认执行软调节逻辑。
- 坏市场专项若重启，主目标固定为“坏市场绝对收益”，不得再用坏市场超额替代坏市场盈利。

## 4. 当前优先级
### 优先级 A：把 shortlist 升级为攻守控制器
目标：

- 在执行端已切回当前仓安全桥接后端的前提下，继续保留当前代码口径的研究主线；
- 不再扩新候选；
- 把已经完成的 shortlist head-to-head 收口成可正式回测的攻守切换规则；
- 且必须以“不低于旧收益前沿”为前提。

具体动作：

1. 只比较以下两组最终候选：
   - `trend_up_low_vol_ml25_none25_v250 @ 504 / 21 / 520`
   - `trend_up_low_vol_ml25_none20_v255 @ 504 / 21 / 520`
2. 比较口径继续固定为：
   - 历史滚动高流动性股票池
   - `next_open`
   - 多窗口 walk-forward
   - `weak_window_20250905_20260319`
3. head-to-head 已完成；当前两组候选的职责固定解释为：
   - `v255` 负责进攻
   - `v250` 负责防守
4. 下一步正式工作不再是“二选一”，而是补一个同口径 formal comparator，比较：
   - 静态 `v255`
   - 静态 `v250`
   - 动态攻守控制器
5. 在攻守控制器正式跑完前，不允许研究侧局部高收益候选静默替换默认值；
   同时也不允许只靠“更稳”把更低收益方案继续升级为默认值。
6. 当前若继续推进动态控制器，顺序更新为：
   - 保留 `benchmark_ret_10d` 作为第二代控制轴，不再退回到只有 `trend_gap + annual_vol` 的首轮规则；
   - 继续围绕 `gap>=0.024192, vol<=0.176128, ret10>=0.014717 / 0.003449` 这类候选补 `full_excess_sharpe`；
   - `focus_streak` 暂保留为解释变量，不直接升格为正式 gating 入口。

### 优先级 B：维持当前执行主线稳态
目标：

- 保持当前默认执行链路可用、可复核、可回滚。

具体动作：

1. 继续依赖 `latest_ml_model.json` 的验证摘要与模型新鲜度保护。
2. 继续把执行端默认值明确解释为：
   - `advanced_ml (ma50 baseline, lgbm) + liquid500 + next_open`
3. 若后续继续做执行端长窗口正式复核，优先沿当前 shortlist 收口，不回退到更早已淘汰参数。
4. 若重启坏市场专项，统一按“坏市场绝对收益”排序与验收。

### 优先级 C：完成 `deep_alpha` 的正式判决
目标：

- 把 `deep_alpha` 的正式能力边界跑清楚；
- 在不打断执行端升级判决的前提下，继续维持主研究线推进。

具体动作：

1. 固定正式框架：
   - 历史滚动高流动性股票池
   - `next_open`
   - 多窗口 walk-forward
2. 先解决关键窗口 `undertrained`，再讨论结构升级。
3. 只有在多窗口持续改善且不破坏既有强窗口时，才允许讨论接近执行端。

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

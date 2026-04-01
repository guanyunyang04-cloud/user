# Daily Research 项目地图

## 1. 用途
本文档是 `daily_research` 的协作导航页，供你和 Codex 在进入具体工作前快速对齐以下问题：

- 项目为什么会演化成现在这条主线；
- 已经解决了哪些核心问题；
- 当前真正的瓶颈是什么；
- 接下来最值得投入的研究方向是什么。

本文档不承载每日默认值、最新 shortlist 或原始实验证据。对应的单一职责仍保持为：

- `daily_research/brain/semantic_memory.md`
  - 当前状态、稳定主线与脑模块边界
- `daily_research/brain/working_memory.md`
  - 当前默认决策、升级 shortlist、优先级与停止规则
- `daily_research/brain/environment_model.md`
  - 解释器、依赖与运行口径
- `daily_research/brain/action_system.md`
  - 执行流程与日常操作细节
- `daily_research/brain/episodic_memory.md`
  - 时间顺序实验记录、证据与结论

## 2. 一句话概括项目
项目已经解决了“如何持续运行一条可信、可执行、可回滚的日线研究与执行链路”，但还没有解决“如何稳定地产生一个跨窗口、跨状态、跨市场环境都优于当前执行主线的新 alpha”。

当前执行默认主线仍是：

- `advanced_ml (ma50 baseline, lgbm) + liquid500 + next_open`

## 3. 立项以来的主线演化
### 阶段 A：从简单规则与打分体系起步
- 起点是全A范围内的简单因子组合、分数加权与调仓频率对比。
- 这一阶段很快收敛出两点：
  - `score` 明显优于 `equal`
  - `5d` 明显优于 `1d`
- 同时也暴露出第一个大问题：
  - 单一全局配方对市场环境高度敏感，强年份和弱年份表现撕裂。

### 阶段 B：从全局规则转向市场状态过滤
- 项目随后转向：
  - 市场状态门控
  - 象限划分
  - 状态白名单
  - 状态专属 profile
- 这一阶段的核心贡献不是“找到一个永远最强的配方”，而是确认：
  - 策略必须条件化；
  - 不能再把不同市场状态当成同一问题处理。

### 阶段 C：把研究口径收敛成可执行链路
- 后续主线从“研究好看”继续推进到“更接近真实执行”：
  - 盘后生成信号
  - 次日开盘执行
  - 历史滚动高流动性股票池
  - 研究端与执行端显式分层
- 这一阶段的关键意义是：
  - 项目不再只比一堆离线回测曲线，而是建立了可直接服务执行端的链路。

### 阶段 D：`advanced_ml` 成为执行主线，`deep_alpha` 成为研究主线
- 在统一到 `next_open` 口径后，`advanced_ml` 逐渐承担正式执行职责。
- 同时期打开了 `deep_alpha` 表示学习主线，试图从：
  - 因子和规则主导
  - 转向更强的时序表示和排序学习。
- 这一步带来了更大想象空间，也带来了真正的新瓶颈：
  - `deep_alpha` 在高流动性局部样本有亮点，但还没有证明自己能稳定泛化到正式执行级别。

### 阶段 E：从“扩分支”转向“收口与治理”
- 3 月下旬之后，项目明显从“快速开分支”切到：
  - 正式框架
  - 多窗口 walk-forward
  - 文档分工
  - 缓存归档
  - 研究结论可回溯
- 当前整体风格已经不是“继续暴力扫参数”，而是：
  - 收敛默认执行主线
  - 收敛升级 shortlist
  - 给 `deep_alpha` 做真正的正式判决

## 4. 到目前为止已经解决的问题
### 4.1 研究与执行的假设统一了
- 当前已经统一到：
  - `next_open`
  - 盘后信号
  - 次日开盘执行
- 这避免了研究端和执行端口径各说各话。

### 4.2 执行主线已经稳定存在
- 当前默认执行主线不是概念性的“AI 选股”，而是非常具体的：
  - `advanced_ml (ma50 baseline, lgbm) + liquid500 + next_open`
- 这意味着项目已经有一条可维护、可运行、可解释的生产主线。

### 4.3 正式研究框架已经建立
- 当前策略晋级必须经过：
  - 历史滚动股票池
  - `next_open`
  - 多窗口 walk-forward
  - 强窗口不被破坏
- 这一步实际上解决了过去量化研究里最常见的问题：
  - 单窗口幸运结果被误当成主线结论。

### 4.4 执行升级问题已经从开放搜索收敛到有限比较
- 最近几轮正式实验已经把执行升级收敛到两组候选：
  - `trend_up_low_vol_ml25_none25_v250 @ 504 / 21 / 520`
  - `trend_up_low_vol_ml25_none20_v255 @ 504 / 21 / 520`
- 而且这两组候选的 formal head-to-head 已经完成。
- 这说明项目当前不是“毫无方向”，而是已经进入“如何把 split verdict 变成可执行决策”的阶段。

### 4.5 坏市场专项的目标函数被纠正了
- 当前已经明确：
  - 坏市场超额不等于坏市场盈利
  - 若重启坏市场专项，目标必须写成坏市场绝对收益或明确的进一步降损
- 这解决了过去很容易发生的表述漂移。

## 5. 当前真正的瓶颈
### 5.1 `deep_alpha` 的核心瓶颈是排序稳定性，不是想法不够多
- 当前最重要的问题不是模型不够复杂，而是 `trend_up_low_vol` 这类关键状态里的 winner-picking 稳定性不足。
- 弱窗口失守主要集中在 `neutral_mixed / pullback_rebound / trend_breakout`。
- 这说明它的问题是状态和结构耦合下的排序质量，而不是单纯的流动性或容量问题。

### 5.1.1 当前最重的工程时间瓶颈是 per-window 全流程重训
- 最近完成的 `score_head / ranking` formal 阶段表明，最耗时的不是 `run_minimal_matrix.py` 的矩阵编排，而是每个 walk-forward window 里的 `run_deep_alpha_research.py`。
- 时间主要花在 `[4/8] Building sequence features and targets` 与 `[6/8] Training deep alpha model`。
- 因此后续研究要把“少开无效分支”和“尽量复用 per-window cache”同时当成默认纪律，优先复用 feature cache / sequence corpus cache / 已落盘 encoder artifact，避免对只改 head 或 ranking 的实验重复走整条慢链路。

### 5.2 执行端的瓶颈已从“缺少控制器”转成“当前机会集天花板过低”
- 当前默认主线并不是被新方案全面推翻，而是已经把 `v250`、`v255` 和双 profile 控制器的 frontier 基本画清。
- `2026-03-29` 的 cross-profile formal comparator 已明确给出三元边界：
  - 静态 defense `expanded_v24 + v250`：`full_annual_return = 13.92%`，`full_excess_sharpe = 0.759`
  - 静态 offense `legacy_v7 + v255`：`full_annual_return = 15.54%`，`full_excess_sharpe = 0.860`
  - 最强 cross-profile dynamic：`full_annual_return = 15.49%`，`full_excess_sharpe = 0.836`，但 `weak_excess_sharpe = 1.359`，`focus_weak_excess_sharpe = 1.458`
- 这说明：
  - 控制器确实能把弱窗口和 focus-weak 防守抬起来；
  - 但它没有把年化前沿推过静态 offense，也没有形成一个同时压过两组静态控制的单一赢家。
- 所以当前真正的短期瓶颈已经不再是“怎么把 offense edge 和 defense edge 拼起来”，而是：
  - 在当前 `liquid500 + next_open + advanced_ml_current_code` 机会集里，clean annual frontier 本身就抬不高；
  - 想继续冲更高收益，必须转向新机会集、新 alpha 家族，或新的收益翻译方式。
- 底层状态机本身仍然有价值，但它现在更像“帮助解释与切换”的基础设施，而不是当前收益天花板的决定性瓶颈。

### 5.3 坏市场里还没有成熟的绝对收益 alpha
- 当前执行主线在坏市场里的能力更接近：
  - 少亏
  - 降仓
  - 空仓
- 它有明确防守价值，但还不能被表述成“坏市场稳定盈利方案”。

### 5.4 项目当前最怕的是局部修复破坏全局稳态
- 很多候选能修一个弱窗口，却会：
  - 破坏强窗口
  - 带来季度集中
  - 只在静态最新股票池里好看
- 所以项目真正的约束条件不是算力，而是：
  - 必须找到跨窗口稳态改善，而不是局部脉冲收益。

## 6. 当前正在推进的工作
### 6.1 正式 live 主线保持稳定，升级候选已经切换
- 当前正式 live 默认值不变：
  - `advanced_ml_current_code_live_anchor (ma50 baseline, lgbm520 v250) + liquid500 + next_open`
- 但“执行端下一步研究”已经不再是旧的 `v250 / v255 / 动态控制器` 参数空间。
- mainboard-only revalidation 与 exact same-window replay 已经把边界压实：
  - `advanced_ml_live_anchor_samewindow_20260401_formal_r1` 的 `bridge_full (2025-03-18 -> 2026-03-31)` 只有 `24.34%` 年化、`10.75%` 超额年化、`0.580` 超额 Sharpe
  - 旧 controller shortlist 只能作为历史 frontier map，不再是默认升级主战场
- 当前执行升级真正正在推进的是：
  - `deep_alpha dynamic_graph_v1 -> target_weight 直连桥 -> anchored offset ensemble`

### 6.2 “翻译损耗”方向已经证明确实有应用价值
- 这条线的核心结论已经从“值得试”升级成“已被 formal 结果验证有效”：
  - `score -> weight` 旧翻译器边际收益很低
  - `target_weight` 直连比旧翻译器更对路
  - 真正的大头来自 `target_weight` 直连后叠加 `5d / 10d cadence + all-offset ensemble`
- 当前最强 anchored finalists 已落盘：
  - `native_anchor_regoff_k2_10d_ensemble = 52.14% / 32.75% / 2.188 / -7.72%`
  - `native_anchor_regon_k1_10d_ensemble = 52.32% / 32.91% / 1.800 / -9.36%`
- `2026-04-01` 晚间这条执行候选桥也已经产品化到 profile wrapper：
  - `run_research_candidate_backtest.py` / `run_research_candidate_trade_plan.py` 现在支持 `--candidate-profile`
  - 默认 profile 直达 `regoff_k2_10d_ensemble_native_anchor`
  - `aggressive` alias 直达 `regon_k1_10d_ensemble_native_anchor`
- `2026-04-01` 深夜又把这条桥推进了一步：
  - `regoff_k2 vs regon_k1` 的 multi-window H2H 已确认两者分工稳定：`regoff_k2` 拿 `4/5` 个窗口的 excess Sharpe，`regon_k1` 拿 `4/5` 个窗口的 excess annual return
  - `robust_composite + train_eval_window_days=252` 的 execution-alignment formal 又选出了新的 high-upside winner：`execalign_auto_r4_topk2_1d_regoff = 108.05% / 85.32% / 3.317 / -11.01%`
  - 这条新候选已产品化为 `robust_auto` alias，但因为换手和回撤压力更大，当前仍不静默替换默认生产候选
- 这里真正重要的新工程边界是：
  - all-offset ensemble 必须固定 `rebalance_anchor_date`
  - 单个 lucky offset 不得直接按生产候选解释
  - `weeklyized_return` 只保留为辅读数，不替代年化主判据
- 同时也已经明确：
  - soft state-conditioned sizing 是风险塑形工具
  - 它当前没有抬高 execution frontier，不是默认执行新主线

### 6.3 `deep_alpha` 已经成为当前新 alpha 家族主前沿
- 这条线不再停留在“潜力分支”，而是已经完成一轮更严格的正式判决。
- strict rolling `liquid800` 在修正窗口 bug 后，旧的 `plain/ranked` 双前沿叙事已经结束；当前 research leading branch 是 `dynamic_graph_v1`。
- 已知收口结论：
  - 原始 `mamba` 在新 universe 下首轮失败，不再作为默认优化对象
  - 旧 `relation_layer` 已失利，不再回到轻量 relation map 小修小补
  - `dynamic_graph_v1` 已正式跑赢旧 `relation baseline` 与当前 `plain`
  - `2026-04-01` 的 formal ablation matrix 又进一步确认：`dynamic_graph_v1` 仍是均值超额年化 winner，`dynamic_graph_no_priors` 几乎贴住它且回撤更浅，`dynamic_graph_topk4` 拿到更高均值超额 Sharpe 但丢失最后一个窗口，`dynamic_graph_topk12` 明显退化
  - `2026-04-01` 晚间的 execution bridge head-to-head 再补了一道 gate：`dynamic_graph_no_priors / dynamic_graph_topk4` 虽然在 `liquid500` research holdout 上都强于 `dynamic_graph_v1`，但在当前默认 `regoff_k2_10d anchored all-offset` execution bridge 下仍未跑赢 `dynamic_graph_v1 = 52.14% / 32.75% / 2.188 / -7.72%`
  - `2026-04-01` 深夜的 execution-objective alignment formal 进一步确认：train-side auto scan 虽然选中了 `regon_k1_10d_ensemble_native_anchor`，但 aligned export replay 只有 `37.26% / 19.76% / 1.145 / -11.56%`，仍输给固定 `regoff_k2` profile 的 `50.89% / 31.66% / 2.160 / -8.59%`
- 因此当前研究线真正正在推进的是：
  - `dynamic_graph_v1` 的稳健性确认
  - 图先验 / 图参数消融
  - 与 execution objective 的进一步对齐
- `2026-04-01` 晚间，这条后续研究也已经产品化到 `run_dynamic_graph_ablation.py`，后续可直接按内置 profile 跑 `plain baseline / winner / top-k / no-prior` 几组关键消融，而不再手抄长命令。

## 7. 接下来最值得投入的研究方向
### 7.1 第一优先级：把 anchored execution candidate 做成正式候选
- 当前最值得投入的，不是回去细磨 `v250 / v255 / controller`，而是把已经正式领先的 execution candidate 收成稳定候选。
- 默认推进顺序应固定为：
  - `regoff_k2_10d_ensemble_native_anchor` 作为默认生产候选
  - `execalign_auto_r4_topk2_1d_regoff` 作为当前最强 formal upgrade shortlist
  - `regon_k1_10d_ensemble_native_anchor` 作为更激进收益对照
  - exact same-window formal comparator 作为升格门槛
- 这条线的研究重点不是“再找一个 lucky offset”，而是：
  - phase robustness
  - anchored cadence 稳定性
  - candidate 计划链路与 formal 回测的一致性
  - 高收益候选的换手 / 滑点现实性

### 7.2 第二优先级：继续沿 `target_weight` 直连桥降低迁移损耗
- 当前已经确认方向是对的，但桥接表达仍未完全做到头。
- 默认后续不再把旧 `score -> weight` 翻译器当主线，而是继续沿：
  - `target_weight` 直连
  - execution objective 对齐
  - cadence / ensemble 工程化
- 这里的判断标准仍然是：
  - 年化收益
  - 超额年化
  - Sharpe
  - 回撤
- `weeklyized_return` 只保留为辅指标，不改变主 verdict。
- soft state-conditioned sizing 只在它能同时抬升前沿时才有资格升格；在此之前，它只是风险塑形 comparator。

### 7.3 第三优先级：让 `dynamic_graph_v1` 继续承担研究主前沿
- `dynamic_graph_v1` 已经正式通过：
  - `relation baseline`
  - 当前 `plain`
- formal ablation matrix `dynamic_graph_ablation_formal_20260401_r1` 进一步把这件事写实了：
  - 图结构本身不是靠 prior 硬撑起来的，`no_priors` 仍能保住大部分前沿
  - `top_k=4` 更像风险收益比对照，不是新的总收益默认 winner
  - `top_k=12` 已可视为 no-go 宽图区间
- execution bridge head-to-head `dynamic_graph_execution_bridge_h2h_20260401_r1` 又补完了第二层筛选：
  - `dynamic_graph_v1` 在默认 `regoff_k2_10d anchored all-offset` execution bridge 下仍是 winner
  - `dynamic_graph_no_priors / dynamic_graph_topk4` 保留为结构与风险收益对照，不升格为默认 execution candidate
- execution objective alignment 已完成两轮 formal：
  - 第一轮 `dynamic_graph_execution_objective_alignment_20260401_r1` 按 `excess_annual_return` 的 auto selector 没有在 aligned export replay 上打赢 `regoff_k2`
  - 第二轮 `deep_alpha_liquid500_dynamic_graph_v1_execalign_auto_20260401_formal_r4` 在 `robust_composite + train_eval_window_days=252` 下选出 `topk2_1d_regoff`
  - `execution_candidate_multiwindow_h2h_regoff_k2_vs_execalign_auto_r4_20260401_r1` 显示：它在 `5/5` 个窗口赢 excess annual return，在 `4/5` 个窗口赢 excess Sharpe
  - 后续若继续做这条线，必须以 aligned export replay 和 multi-window H2H 为主判据
- 所以下一步不应回到旧的 `plain/ranked/controller` 或原始 `mamba` 小修小补，而应继续做：
  - 只在 aligned export replay + multi-window H2H 口径下继续做 `dynamic_graph_v1` 的 execution objective 对齐
  - 优先复核 `execalign_auto_r4_topk2_1d_regoff` 的高换手现实性，而不是回头调旧翻译器
  - 若需要对照，只保留 `dynamic_graph_no_priors / dynamic_graph_topk4` 两条 reference branch
- 若这条线最终不能稳定守住前沿，再切到 `state-conditioned MoE`；RL 仍只保留在执行层独立分脑。

### 7.4 第四优先级：若重启坏市场专项，必须单独立题
- 这条线只有在目标明确写成：
  - 坏市场绝对收益
  - 或更进一步的坏市场降损
  时才值得继续投资源。
- 它不能再作为上涨态微调的副产品存在。

### 7.5 第五优先级：继续强化项目治理，而不是回退到口头管理
- 这个项目后续能否长期维持质量，很大程度取决于：
  - 文档边界
  - 研究结论可回溯
  - 热区治理
  - 代理协同前的上下文对齐

## 8. 明确应降级或停止的方向
- `base_global` 参数优化：
  - 可以保留为对照，但不再是执行升级主候选
- `504 / 5 / 260`
  - 仍有对照价值，但已退出优先晋级序列
- `378 / 21 / 520`
  - 未通过长窗口正式复验，停止继续推进
- `ma47/48` 左侧边界带
  - 保留在研究附录，不再占主带宽
- `v21_volume_contraction_015`
  - 保留在规则层研究附录，不进入默认执行口径
- 连续状态软调节
  - 保留为诊断层，不进入默认执行逻辑
- 任何只改善坏市场超额、却不能改善坏市场绝对收益的方案
  - 不再按坏市场盈利方案解释

## 9. 推荐协同阅读顺序
后续无论是你还是 Codex，建议统一按这个顺序建立上下文：

1. 先看 `daily_research/brain/semantic_memory.md`
   - 确认当前状态、主入口与文档边界
2. 再看 `daily_research/brain/project_map.md`
   - 快速理解项目背景、当前瓶颈和未来方向
3. 再看 `daily_research/brain/working_memory.md`
   - 确认当前待决策事项和优先级
4. 需要落命令时看 `daily_research/brain/environment_model.md`
   - 确认解释器和运行口径
5. 需要执行链路时看 `daily_research/brain/action_system.md`
   - 确认盘后操作流程和执行边界
6. 需要追溯证据时查 `daily_research/brain/episodic_memory.md`
   - 看完整时间线和实验结论来源

## 10. 协作纪律
- 不在本文件里追加按日期写的实验日志。
- 不在本文件里偷偷写新的默认参数或最新执行结论。
- 本文件只在以下情况更新：
  - 项目主线发生阶段性转移
  - 当前瓶颈判断发生明显变化
  - 未来研究方向发生实质性重排
  - 协作入口和阅读顺序发生变化

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
### 6.1 执行端升级 shortlist 的最终判决
- 这是当前最靠近真实决策的任务。
- 当前 formal head-to-head 已完成，比较对象固定为：
  - `trend_up_low_vol_ml25_none25_v250 @ 504 / 21 / 520`
  - `trend_up_low_vol_ml25_none20_v255 @ 504 / 21 / 520`
- 当前已知分工：
  - `v250` 更偏弱窗口稳健性
  - `v255` 更偏全样本收益进攻性
- 当前直接结论：
  - 旧 formal 结论已经因 `market_features` 扩容和 rolling ML score 漂移而失效；
  - 当前代码下，`v250` 已经成为执行端 live 默认值；
  - `v255` 降级为进攻对照，不再作为默认值候选。
- 当前更合理的下一步：
  - 不再追问旧 split verdict 到底谁赢；
  - 先用双 profile 控制器把 frontier 画清，再决定是否值得继续留在当前参数空间。
- 当前的 formal R3 已经把边界更新成了：
  - 静态 `v250 = 0.759 / 1.073 / 1.083`
  - 静态 `v255 = 0.675 / 0.281 / 0.094`
  - 最强动态候选 `= 0.791 / 0.992 / 0.980`
  - 所以动态方向仍成立，但研究 benchmark 已改成 live `v250` 默认值。
- `market_feature_profile_compare_20260328_formal_r1` 又把更深一层的边界补清了：`expanded_v24` 修复了 `base_global`、让 live `v250` 的弱窗口更稳，但也明显吃掉了旧 `v255` 与当前动态控制器的进攻上沿。
- `market_feature_profile_pruning_20260328_formal_r1` 继续把边界压实了：几组中间态 profile 都没能同时保住当前 `expanded_v24` 的 live 防守、又恢复旧 `legacy_v7` 的进攻上沿；最接近的是 `continuous_quadrant_v9`，但它仍然是“弱窗口修回来一些、full 端和 live `v250` 都变差”。
- `advanced_ml_cross_profile_attack_defense_20260329_formal_r1` 已经把这件事正式跑完：
  - best dynamic `15.49% / 0.836 / 1.359 / 1.458`
  - static offense `15.54% / 0.860 / 0.819 / 0.751`
  - static defense `13.92% / 0.759 / 1.073 / 1.083`
- 因此当前执行端的真实瓶颈已进一步收敛为：不是“控制器还没做”，而是“控制器做完后，当前机会集的 clean annual frontier 仍然不够高”。
- 用户已在 2026-03-28 明确纠偏：坏市场、弱窗口、稳定性都属于从属目标，它们必须服务于“总利润继续抬高”，而不是替代总利润目标本身。
- 这意味着项目的下一阶段不该再把“更稳但更低收益”解释成自然升级，而应回到更原始也更严格的目标：
  - 每个阶段都追求最大利润
  - 不同阶段可以有不同策略
  - 但阶段切换必须提高总利润，而不是牺牲旧收益前沿换局部稳定性
### 6.2 保持执行主线稳态
- 当前仍在持续维持：执行链路可运行、模型新鲜度保护、研究结果不能静默替换默认值。

### 6.3 `deep_alpha` 的正式判决没有结束
- 这条线没有被放弃，但目标已经收缩为：先解决关键窗口 `undertrained`、先修关键状态下的排序稳定性、先证明多窗口一致性。
- `2026-03-29` 这条线已经跑完整个最小矩阵：`deep_alpha_minimal_matrix_20260329_backbone_r1` 的 `backbone / score_head / ranking` 都已完成，winner 始终是 `patch_transformer + no pretrain + manual + plain`。
- `masked pretrain`、`ridge / lgbm` score head、`ranked` loss 都没有把前沿抬高。
- 这意味着“新 alpha 家族”已经完成第一轮最小充分判决，下一步不该继续深挖这几个已输掉的小旋钮，而要把资源投向真正改变机会集或表示能力的分支。

## 7. 接下来最值得投入的研究方向
### 7.1 第一优先级：转向新机会集 / 新 alpha 家族
- 当前最值得投入的，不再是继续细磨 `v250 / v255 / controller`，因为 formal cross-profile comparator 已证明：
  - 控制器能补稳健；
  - 但不能把年化前沿抬出新台阶。
- 下一步真正值得做的，是寻找新的收益来源，而不是继续在当前 frontier 附近绕圈：
  - 新 universe / 新容量约束
  - 新 alpha family
  - 新组合构建与收益翻译方式
- 这些新方向仍必须沿用当前正式纪律：
  - 历史滚动股票池
  - `next_open`
  - 多窗口 walk-forward
  - 不破坏强窗口

### 7.2 第二优先级：让 `deep_alpha` 从“局部有效”走向“正式可判决”
- 值得投入的不是泛化的“更大模型”，而是更具体的三件事：
  - 强化 `trend_up_low_vol` 下的排序稳定性
  - 优先修复弱窗口里的关键结构
  - 继续用 rolling liquid pool + `next_open` + 多窗口 walk-forward 验证
- 当前最小矩阵已经证明：沿 `deep_alpha_minimal_matrix_20260329_backbone_r1` 继续细磨 `score_head / ranking` 的边际收益很低。
- 下一步更值得投入的是新机会集、新 encoder / 表示能力，或更彻底的收益翻译方式。

### 7.3 第三优先级：保持当前执行主线稳态
- 当前执行默认值仍应继续停在：
  - `advanced_ml_current_code_live_anchor (ma50 baseline, lgbm520 v250) + liquid500 + next_open`
- cross-profile controller 的 formal 结果应只作为研究 frontier map 使用，不应再触发默认值来回摇摆。

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

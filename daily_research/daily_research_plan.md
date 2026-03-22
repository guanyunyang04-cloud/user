# Daily Research 研究总纲

## 1. 文档分工
- `README.md`
  - 面向使用。
  - 说明当前框架能做什么、怎么运行、输出什么。
- `research_log.md`
  - 面向研究记录。
  - 按时间沉淀实验、参数、结果、结论。
- `daily_research_plan.md`
  - 面向规划与主线。
  - 说明研究目标、当前结论、下一步路线。

三者关系：
- `README.md` 告诉我们“怎么用”。
- `research_log.md` 告诉我们“做过什么、结果如何”。
- `daily_research_plan.md` 告诉我们“接下来做什么、为什么做”。

## 2. 当前研究目标
- 构建日线级别的选股与仓位研究框架。
- 形成可回测、可解释、可迭代的研究流水线。
- 在全A范围内，围绕“相对基准超额”持续优化。

当前阶段边界：
- 频率：日线
- 成交假设：当日收盘调仓成交
- 股票池：全A
- 组合风格：高集中
- 评估标准：组合收益、基准收益、超额收益、因子 IC / RankIC、分层收益
- 暂不纳入：
  - 机器学习
  - 强化学习
  - 分钟级执行
  - 行业中性化
  - 基本面因子

## 3. 当前框架主线

### 3.1 数据层
- TQ 模式：
  - 通过 `baseline/data_provider.py` 调用 `t0_project/tqcenter.py` 获取全A和基准数据。
- CSV 模式：
  - 用本地文件做离线复现。

### 3.2 因子层
当前主线已收敛到以下风格：
- 量能：
  - `volume_contraction`
  - `price_volume_divergence`
- 波动：
  - `atr_14_pct`
  - `volatility_20`
  - `volatility_contraction`
- 结构：
  - `close_strength`

已知结论：
- 趋势组整体不适合作为全局默认主线。
- 结构组不能完全删除。

### 3.3 组合层
- 默认是高集中组合。
- 已验证 `score` 加权优于 `equal`。
- 已验证 `5d` 调仓优于 `1d`。
- 已验证高集中优于过度分散。

### 3.4 状态层
当前研究已经从“单一全局因子权重”进化到“市场状态 + 状态内权重切换”。

市场状态四象限：
- `trend_up_low_vol`
- `trend_up_high_vol`
- `trend_down_low_vol`
- `trend_down_high_vol`

当前主过滤逻辑：
- 规避两个下行象限
- 允许两个上涨象限：
  - `trend_up_low_vol`
  - `trend_up_high_vol`

### 3.5 状态内动态权重
当前最强候选 profile：
- `up_low_breakout_v2`

含义：
- `trend_up_low_vol`
  - 用“更克制的低波 + 结构增强”
- `trend_up_high_vol`
  - 当前仍以基线处理为主，高波增强继续保留为实验分支

## 4. 已有关键结论

### 4.1 已被验证成立
- `score + 5d` 明显优于 `equal + 1d`
- 市场状态过滤有效
- 行业上限不是当前收益提升的关键来源
- 风格约束有小幅帮助，但不是根本解
- `2025` 的问题不是没有 alpha，而是上涨低波环境下组合过于防守
- 状态内动态权重能显著改善 `trend_up_low_vol`
- `trend_up_high_vol` 也值得单独建模

### 4.2 当前仍未解决
- `up_dual_v1 / up_dual_v2` 虽然能改善部分样本，但在冻结验证和激活验证下仍不够稳健
- `up_low_breakout_v2` 是当前最接近“可接受增强版本”的状态内 profile
- 也就是说，profile 方向大概率是对的，但“事前激活机制”还不够好

## 5. 当前主线配置
当前建议作为研究主线的配置是：
- 股票池：全A
- 基准：`000300.SH`
- 权重：`score`
- 调仓频率：`5d`
- 市场状态过滤：开启
- 市场状态白名单：
  - `trend_up_low_vol`
  - `trend_up_high_vol`
- 风格约束：开启
  - `max_style_weight=0.50`
- 状态内动态权重：
  - 稳健基线：`none`
  - 增强候选：`up_low_breakout_v2`
- 事前激活候选：
  - `trend_up_high_vol`：固定 `none`
  - `trend_up_low_vol`：在 `none` 与 `up_low_breakout_v2` 之间做 `20d RankIC` 的季度切换
  - 训练窗口：优先 `24` 个月
  - 当前规则候选：
  - `breakout_20 RankIC >= 0`
  - `drawdown_20 RankIC >= 0`
  - `range_position_20 RankIC >= -0.06`

## 5.1 交付导向升级
为了提高交付效率，当前新增一条并行主程序线：
- `daily_research/baseline/run_advanced_daily_research.py`

目标：
- 不再只依赖手工因子加权
- 在现有框架上叠加机器学习横截面排序
- 尽快形成一套更完整、更强的可运行程序

当前方案：
- 机器学习模型：`sklearn` 梯度提升树
- 预测目标：未来 `20d` 超额收益
- 集成方式：
  - 机器学习分数
  - `none`
  - `up_low_breakout_v2`
- 继续保留：
  - 市场状态过滤
  - 高集中组合
  - 风格约束

## 6. 下一步研究优先级

### 优先级 A：事前激活机制
目标：
- 让增强版本在冻结验证与事前激活中更稳定，而不是只在全样本后验里好看。

建议方向：
1. 在先进版主程序上先跑全A正式实验
2. 用连续回测口径重做 RankIC 激活对照
3. 对规则候选做阈值稳健性验证

### 优先级 B：状态内再细分
目标：
- 继续优化上涨双象限内部的权重切换。

建议方向：
1. `trend_up_low_vol` 内部再分层
2. `trend_up_high_vol` 内部再分层

### 优先级 C：风险与执行
目标：
- 在不破坏当前 alpha 的前提下，提升稳健性。

建议方向：
1. 更细的风格暴露约束
2. 波动目标或回撤控制
3. 后续再考虑连接日内执行层

## 7. 里程碑视图
- 已完成：
  - 第一阶段框架搭建
  - 因子收敛
  - 市场状态过滤
  - 上涨双象限白名单
  - 状态内动态权重原型
- 正在推进：
  - profile 的样本外验证与激活机制
- 后续规划：
  - 更严格的冻结验证
  - 更细的状态切换
  - 连接更完整的仓位与执行研究
## 新阶段主线：Deep Alpha

在当前 `baseline/advanced_ml` 已经形成稳定可用主线之后，研究重点正式扩展为两条并行路线：

### 1. 稳定交付主线
- 继续维护 `baseline/` 与 `execution/`
- 目标是保证：
  - 每日尾盘建议可用
  - advanced ML 主线稳定
  - 只有通过更大样本验证的配置才进入执行端默认值

### 2. 结构理解主线
- 新分支：`daily_research/deep_alpha/`
- 目标不是继续做小幅参数扫描，而是研究更本质的问题：
  - 市场状态是否可以被自动学习
  - 股票历史序列能否学出稳定表示
  - 横截面排序能否从“因子工程”升级到“表示学习 + 多任务预测”

当前 `deep_alpha` 已完成第一版骨架：
- 市场状态学习：`market_state_model.py`
- 序列样本构建：`sequence_dataset.py`
- 时序编码器：`models.py`
- 训练与推断：`trainer.py`
- 统一入口：`run_deep_alpha_research.py`

### 当前结论
- `deep_alpha` 已经可以完整跑通。
- 第一轮烟测中：
  - `Transformer` 优于 `GRU`
  - 但整体仍弱于当前成熟的 `advanced_ml` 主线
- 这说明方向值得继续，但当前阶段重点应放在：
  1. 排序目标设计
  2. 时序表示质量
  3. 预测到持仓分数的映射

### 最新推进
- `deep_alpha` 已经从“逐样本回归”升级到：
  - 按日期分组训练
  - 回归损失 + 排序损失
- 最新烟测表明：
  - 排序损失对 `10d / 20d` 超额 RankIC 有明显正向作用
  - 说明后续应该继续沿“训练目标更贴近横截面排序”推进
  - 而不是回头做传统的小参数微调
- 关系层第一版也已经接入：
  - 行业内排名
  - 行业强度
  - 风格强度
- 但在当前极小烟测样本里，关系层第一版拖累了表现，因此暂不进入默认研究配置。
- 同时，`deep_alpha` 已经新增学习式二层 score head：
  - 不再只靠手工把多任务预测值加总成持仓分数
  - 可以用 `Ridge` 学习“预测 -> 持仓分数”的映射
  - 目前已初步优于手工映射
  - 进一步升级后，`LightGBM score head + 窗口式 adaptive task weights` 已经优于手工映射和 Ridge 版本

### 下一步优先级
1. 在 `deep_alpha` 中引入更贴近横截面排序的训练目标
2. 优化序列特征表达和样本构造
3. 再和当前 `advanced_ml` 主线做更大样本对照

## 2026-03-19 阶段判断与主线收敛

### 当前判断
- `deep_alpha` 已经从“概念验证”进入“稳定研究主线”阶段。
- 当前最值得持续推进的方向已经比较清楚：
  1. 第一层 `return` 学习增强
  2. 数据驱动市场状态学习
  3. 关系层第二版轻实现
- 与之相对，`advanced_ml` 继续承担当前执行端默认主线，不再优先做局部微调。

### 当前 deep_alpha 主线
- `raw return target`
- `pairwise rank loss`
- `listwise rank loss`
- `state_gate`

### 研究纪律
- 不再优先做小权重扫描与 profile 命名扩张。
- 不再用极小烟测结果直接推动执行端配置变更。
- 只有在更大样本、同池同区间对照里持续胜出，才讨论接近执行端。

## 2026-03-20 当前阶段规划更新

### 已完成
1. `deep_alpha` 新主线在 `liquid500` 上完成了更长 `252` 日 holdout 严格复验。
2. `deep_alpha` 新主线在 `liquid800` 更广高流动性子样本上完成了同池同区间对照。
3. `advanced_ml` 已支持 `--stocks-file`，方便后续严格同池对照。

### 当前结论
- `deep_alpha` 在 `liquid500` 和 `liquid800` 两个更大高流动性样本上都赢过了当前 `advanced_ml`。
- 这说明它已经具备“接近执行主线”的候选资格。
- 但当前仍不直接替换执行端默认主线，原因是：
  - 更广子样本还需要继续验证
  - 回撤稳定性还需要进一步确认

### 下一步优先级
1. 做更广高流动性子样本复验，或做分层全A子样本复验。
2. 做执行前最后对照，重点看：
   - 超额收益
   - 超额 Sharpe
   - 超额最大回撤
   - 换手稳定性
   - 胜率与月度一致性
3. 如果优势继续保持，再讨论让 `deep_alpha` 接近执行端。

### 当前主线分工
- `advanced_ml`
  - 当前执行端默认主线
  - 优先保证稳定可用
- `deep_alpha`
  - 当前最有前景的研究主线
  - 目标是做出真正理解市场结构、并能在更大样本里稳定胜出的策略框架
## 2026-03-20 分层全A子样本复验后的规划调整
### 已完成
1. 在 `liquid500` 与 `liquid800` 高流动性样本上，`deep_alpha` 新主线都赢过了当前 `advanced_ml`。
2. 进一步做了分层全A子样本复验：
   - 5 档流动性分层
   - 每档等量抽样
   - 共 `800` 只股票

### 新结论
- `deep_alpha` 并没有在分层全A子样本上保持优势。
- 这说明它当前的超额能力仍明显依赖高流动性环境，而不是已经形成全A范围的稳定泛化。
- 因此，项目当前判断更新为：
  - `advanced_ml` 继续作为执行端默认主线
  - `deep_alpha` 继续作为最重要的研究主线，但暂不接近执行端

### 接下来的研究重点
1. 把“流动性条件”提升为显式研究对象，而不是默认背景条件。
2. 研究 `deep_alpha` 在高流动性样本里有效、在分层全A子样本里失效的结构原因。
3. 优先考虑：
   - 流动性分层状态
   - 流动性相关关系特征
   - 按流动性条件切换 return 学习或 risk gate

### 当前执行决策
- `advanced_ml`
  - 继续保留为执行端默认主线
- `deep_alpha`
  - 不进入执行前最后验证阶段
  - 先回到“更本质的结构学习”路线

## 2026-03-20 流动性研究主线更新

### 当前结论
- `deep_alpha` 在高流动性 `500/800` 股票池上能明显跑赢当前 `advanced_ml`，但在分层全A `800` 子样本上优势消失。
- 这说明 `deep_alpha` 当前最核心的结构性问题，不是“模型完全无效”，而是“对流动性条件高度敏感”。
- 因此，流动性条件已经从背景变量升级为必须显式建模的研究对象。

### 已验证过但暂不作为主线的做法
1. 直接把 `liquidity_layer` 堆进主特征栈。
2. 直接把 `risk gate` 升级成 `state_liquidity_gate`。

当前这两条线在分层全A子样本上都没有带来正向增益，因此不进入默认研究配置。

### 接下来固定的研究顺序
1. 先做流动性失效诊断
- 分析 `deep_alpha` 在不同流动性桶、不同市场状态下的 return 学习质量。
- 重点确认：到底是哪些流动性层拖累了排序质量，还是流动性和状态的交互在拖累结果。

2. 再做“按流动性条件切换 return 学习”
- 不把流动性简单当普通输入特征。
- 优先把流动性作为条件变量：
  - 按流动性层切换 return 目标
  - 按流动性层切换排序损失或 head
  - 让模型学会“在不同流动性环境里，winner-picking 规则不一样”

3. 最后再决定是否重做 liquidity-aware gate
- 只有在 return 学习已经证明有改进后，才重新评估 risk gate 是否需要按流动性条件自适应。

### 当前项目分工不变
- `advanced_ml`
  - 继续作为执行端默认主线
  - 只做稳健性维护，不做大幅配置切换
- `deep_alpha`
  - 继续作为最有前景的研究主线
  - 当前优先级：`return 学习增强 > 数据驱动状态学习 > 关系层第二版轻实现`
  - 现在正式补充为：`流动性条件研究` 穿插在第一层 return 学习增强过程中同步推进

## 2026-03-20 流动性条件化 return 学习后的规划收敛

### 已完成
1. 做了更细的流动性失效诊断，确认 `deep_alpha` 在分层全A子样本里主要只在最高流动性桶里还能维持正向 `fwd_excess_20 RankIC`。
2. 做了第一版“按流动性条件切换 return 学习”：
   - `--return-head-mode liquidity_switch`
3. 在分层全A `800` 子样本上完成了同口径验证。

### 新结论
- “把流动性当条件变量”这个方向是对的。
- 但第一版实现方式——按每个流动性桶切独立 return head——过于生硬，明显拖累了泛化结果。
- 因此，下一步不再沿着“多桶独立 return head”继续深入。

### 接下来更有前景的做法
1. 把流动性条件简化成更少、更稳的 regime
- 例如：
  - `最高流动性`
  - `非最高流动性`

2. 让流动性优先影响 return 学习过程，而不是直接替换 return head
- 优先考虑：
  - 流动性条件化的 return loss 权重
  - 流动性条件化的样本加权
  - 高流动性样本上的 winner-picking 强化

3. risk gate 继续放在第二位
- 目前证据反复说明：问题核心仍然在第一层 return 学习，而不是 risk gate 不够复杂。

### 当前主线不变
- `advanced_ml`
  - 继续作为执行端默认主线
- `deep_alpha`
  - 继续作为研究主线
  - 下一步优先方向更新为：
    - `return 学习增强`
    - `流动性条件化 return 学习`
    - `数据驱动状态学习`

## 2026-03-20 执行与研究假设统一

### 为什么要改
- 之前执行端默认是“同日收盘生成建议、同日尾盘执行”，研究端也有大量 close-to-close 假设。
- 这会带来两个现实问题：
  1. 盘中与收盘的同日缓存可能混用，容易把未收盘数据误当成最终日线。
  2. 实盘上更自然的流程其实是：盘后拿到完整日线，再在次日开盘执行。

### 现在统一成什么
- `advanced_ml` 的训练、推理、研究回测，统一切到：
  - `盘后生成信号`
  - `次日开盘执行`

### 已落地
1. 训练端与推理端现在默认只使用“最新已完成交易日”的日线，不再把当日未收盘 bar 作为最终信号输入。
2. 执行端计划文本改成：
   - `信号日期`
   - `执行日期`
   - `盘后生成、次日开盘执行`
3. `advanced_ml` 训练目标和研究回测已开始按 `next_open` 口径对齐。

### 后续约束
- 只要是准备迁移到执行端的研究结论，都必须优先用 `盘后信号 -> 次日开盘执行` 口径验证。

### 当前状态补充
- `advanced_ml`
  - 执行端与训练链路已经切到 `next_open`
  - 仍然保留执行端默认主线地位
- `deep_alpha`
  - 研究回测也已切到 `next_open`
  - 后续所有与 `advanced_ml` 的对照，默认都用同一 `next_open` 口径

## 2026-03-20 全A next_open 正面对照后的规划更新

### 已完成
- 已完成 `deep_alpha` 与 `advanced_ml` 的全A `next_open` 正面对照。
- 对照输出：
  - `daily_research/output/advanced_ml_alla_nextopen_compare_20260320`
  - `daily_research/output/deep_alpha_alla_raw_listwise25_e4_v252_nextopen_20260320`
  - `daily_research/output/deep_alpha_vs_advanced_ml_alla_nextopen_20260320.csv`

### 当前结论
- 在同一全A、同一 `next_open` 口径下：
  - `deep_alpha` 没有保持住此前在高流动性子样本里的优势。
  - `advanced_ml` 虽然这一轮全A holdout 也没有跑赢基准，但整体仍优于 `deep_alpha`。
- 因此：
  - `advanced_ml` 继续保留为执行端默认主线。
  - `deep_alpha` 暂不进入执行前最后验证阶段。

### 对 deep_alpha 的直接含义
- 当前最重要的问题进一步收敛为：
  - `deep_alpha` 的有效性仍明显依赖高流动性环境。
  - 泛化到更广全A范围时，第一层 `return` 学习依旧不够稳。

### 接下来的固定研究顺序
1. 继续强化 `deep_alpha` 第一层 `return` 学习。
2. 把流动性当条件变量，而不是普通特征。
3. 优先解释并修复“为什么只在高流动性环境里有效”。
4. 只有在更广样本上重新建立优势后，才重新讨论它是否接近执行端。

## 2026-03-20 流动性条件化 return 学习后的规划更新

### 已验证
- 已验证“`top_liquidity` vs `other`”的第一版条件化 return 学习。
- 这版不再切独立 head，而是直接影响：
  - `return loss`
  - `ranking loss`
  - `sample weighting`

### 当前结论
- 这条线对高流动性样本是有效的：
  - `liquid500` 上，条件化版本显著强于 `shared baseline`
- 但它还没有把分层全A泛化彻底拉回来：
  - `stratified_all_a_800` 上，最好的“轻条件化”仍然是负超额
- 因此目前的判断是：
  - 方向正确
  - 但还处在“能增强高流动性进攻能力、尚未修复广样本泛化”的阶段

### 接下来的固定研究顺序
1. 保留“流动性作为条件变量”这条主线。
2. 优先做更平滑的条件化 loss，而不是继续粗暴放大权重。
3. 继续拆开分析：
   - 高流动性里哪些结构被强化了
   - 非高流动性里哪些结构仍然在拖后腿
4. 只有当分层全A子样本显著改善后，才重新评估 `deep_alpha` 是否重新接近执行端。

## 2026-03-20 更平滑流动性条件化后的规划更新

### 新增结论
- 已完成 `smooth_bucket` 版流动性条件化复验。
- 结论不是“越平滑越好”，而是：
  - `smooth_bucket` 没有自动带来更强泛化
  - 它在 `stratified_all_a_800` 上只做到了小幅修复，仍未转正
  - 在 `liquid500` 上反而明显弱于此前的轻度 `top_liquidity conditioning`

### 结构层面的新认识
- 高流动性里真正被强化的是：
  - `trend_breakout`
  - `high_vol_expansion`
  - 次之是 `pullback_rebound`
- 非高流动性里真正拖后腿的是：
  - `neutral_mixed`
  - `pullback_rebound`
  - `low_vol_trend`
  - `weak_structure`

### 这意味着什么
- 当前矛盾已经进一步收敛：
  - 我们不是还没找到“高流动性里有效的结构”
  - 而是还没有学会：
    - 如何只增强这些高流动性进攻结构
    - 同时不破坏非高流动性里的基础排序能力

### 规划调整
接下来不再优先做“继续放大流动性条件权重”，而改成：

1. **定向增强高流动性进攻结构**
- 重点针对：
  - `trend_breakout`
  - `high_vol_expansion`
- 让条件化更多作用在排序/进攻层，而不是粗放地抬高所有高流动性样本权重

2. **保护非高流动性基础结构**
- 优先保护：
  - `neutral_mixed`
  - `pullback_rebound`
  - `low_vol_trend`
  - `weak_structure`
- 目标是避免“为了放大高流动性进攻能力，反而把广样本稳定性打坏”

3. **条件化继续优先作用在第一层 return 学习**
- 但收缩为：
  - 更定向的 ranking/listwise 调整
  - 更少的全局 return loss 放大

### 当前主线不变
- `advanced_ml`
  - 继续作为执行端默认主线
- `deep_alpha`
  - 继续作为研究主线
  - 当前最优先的工作更新为：
    1. 第一层 `return` 学习增强
    2. 定向的流动性条件化排序增强
    3. 非高流动性结构保护与泛化修复

## 2026-03-20 定向结构增强后的规划更新

### 新结论
- 已验证“只对 `top_liquidity` 里的进攻结构做排序增强、同时对 `other` 组基础结构做保护”这条线。
- 结果比 `smooth_bucket` 更好，也比此前的粗放流动性条件化更接近我们真正想要的方向：
  - `liquid500` 上显著增强
  - `stratified_all_a_800` 上虽然仍未转正，但改善幅度继续扩大

### 这意味着什么
- 流动性条件化的有效方式，越来越清楚了：
  - 不是“整体抬高高流动性权重”
  - 而是“定向增强高流动性里的进攻结构”
- 当前最该继续做的，不是更粗暴，而是更精细：
  - 进攻结构增强
  - other 组基础结构保护

### 当前最重要的未解问题
1. `top_liquidity` 里的 `high_vol_expansion` 已经能被放大  
2. 但 `trend_breakout` 还没有被稳定强化出来  
3. `other` 组里：
   - `neutral_mixed`
   - `low_vol_trend`
   - `pullback_rebound`
   仍然是广样本泛化修复的关键

### 接下来的固定研究顺序
1. 继续保留“流动性作为条件变量”这条主线。
2. 不再优先做全局粗暴调权。
3. 下一步优先做：
   - 定向增强 `top_liquidity` 下的 `trend_breakout`
   - 保护 `other` 组里的 `neutral_mixed / low_vol_trend / pullback_rebound`
4. 条件化优先作用在：
   - `pairwise rank loss`
   - `listwise loss`
   - 尽量少动全局 `return loss`

### 当前执行判断不变
- `advanced_ml`
  - 继续作为执行端默认主线
- `deep_alpha`
  - 继续作为最有前景的研究主线
  - 但在分层全A子样本仍未转正之前，不进入执行前最后验证阶段

## 2026-03-20 执行端股票池策略更新：从全A转向高流动性专用池

### 为什么要调整
- 执行规则已经切换成：
  - `盘后信号 -> 次日开盘执行`
- 在这个口径下，旧的全A判断已经不够用了。
- 新的研究结果说明：
  - `advanced_ml` 在更广全A里会被稀释
  - 但在高流动性池里仍然有明显 edge

### 已完成的对照
- 已用 `next_open` 口径重新验证：
  - `liquid300`
  - `liquid500`
  - `liquid800`
- 汇总：
  - `daily_research/output/advanced_ml_liquidity_pool_compare_20260320.csv`

### 当前执行结论
- `liquid300`
  - 不够稳，不适合作为执行端默认池
- `liquid500`
  - 当前盈利能力最强
  - 适合作为执行端默认高流动性基础池
- `liquid800`
  - 更平衡，但略弱于 `liquid500`
  - 可作为保守备选

### 当前推荐模式
执行端不再默认走全A，而是改成：

1. **执行主线**
- `advanced_ml`
- 股票池：`liquid500`
- 口径：`盘后信号 -> 次日开盘执行`

2. **研究主线**
- `deep_alpha`
- 继续聚焦高流动性进攻结构学习
- 不再以“全A通吃”为近期目标

### 接下来的固定研究顺序
1. 执行端围绕 `advanced_ml + liquid500` 稳定化。
2. 研究端继续强化 `deep_alpha` 在高流动性池里的 winner-picking。
3. 只有当 `deep_alpha` 在高流动性专用池里显著长期优于 `advanced_ml`，再讨论接近执行端。

## 2026-03-20 执行端股票池流程固化
- 已新增正式的高流动性股票池更新脚本：
  - `daily_research/execution/update_liquid_pool.py`
- 执行端默认股票池正式改成：
  - `daily_research/execution/universe/liquid500_latest.txt`
- 每日盘后固定流程调整为：
  1. 运行 `update_liquid_pool.py`
  2. 刷新 `liquid300 / liquid500 / liquid800`
  3. 按需运行 `update_model.py`
  4. 每日运行 `run_trade_plan.py`
- 这样做的作用：
  - 让执行端股票池和 `next_open` 实盘口径统一
  - 让高流动性基础池变成正式、可每日更新的流程
  - 同时给 `deep_alpha` 的高流动性专用研究提供统一股票池基线

## Deep Alpha 固定研究池与下一步
### 当前固定研究池
- `deep_alpha` 后续不再默认回到全A泛化主线。
- 固定研究池改成执行端每日更新的高流动性池：
  - `liquid500`：主研究池
  - `liquid800`：扩展验证池
- 研究入口已支持：
  - `--liquidity-pool liquid500`
  - `--liquidity-pool liquid800`

### 最新研究结论
- 在每日更新的固定高流动性池里，当前 `deep_alpha` 的共享基线：
  - `liquid500 shared`
  - `liquid800 shared`
  依然比当前这版 `targeted_liquidity_rank` 更稳。
- 这说明：
  - 高流动性专用研究方向仍然正确
  - 但结构强化不能再粗暴沿用旧参数，必须针对“每日动态流动性池”重新校准

### 接下来的研究顺序
1. 继续保留高流动性专用研究，不再回到全A泛化主线。
2. 以 `liquid500 shared` 为主研究基线，`liquid800 shared` 为扩展验证基线。
3. 下一步结构强化优先做：
   - 更轻的 `trend_breakout` 定向增强
   - 不再同时强推 `high_vol_expansion`
   - 结构条件化继续优先作用在 `pairwise/listwise`，避免直接扭曲全局组合映射
4. 每次新方案都必须同时过：
   - `liquid500`
   - `liquid800`
   两个每日更新固定池的同口径验证

## 2026-03-20 历史滚动高流动性研究框架
### 为什么现在先做这个
- 执行端已经冻结为：`advanced_ml + liquid500 + next_open`
- 研究端如果继续用 `liquid500_latest.txt / liquid800_latest.txt` 静态回看历史，会引入成分前视/生存者偏差
- 所以下一步最高优先级不是继续调模型，而是先把高流动性股票池做成**历史滚动、与当前交易模式严格一致**的正式研究框架

### 已完成的框架能力
- `daily_research/execution/liquidity_universe.py`
  - 新增历史滚动高流动性池构建：
    - `build_rolling_liquidity_membership(...)`
- `daily_research/deep_alpha/run_deep_alpha_research.py`
  - 新增：
    - `--rolling-liquidity-pool liquid300|liquid500|liquid800`
    - `--pool-rebalance-days`
    - `--pool-adv-window`
- `daily_research/baseline/run_advanced_daily_research.py`
  - 新增同口径的历史滚动高流动性池研究入口
- `daily_research/deep_alpha/sequence_dataset.py`
  - 训练语料支持按“历史当期股票池成员资格”过滤样本

### 当前正式研究口径
- 股票池：历史滚动 `liquid500 / liquid800`
- 重建频率：默认每 `21` 个交易日
- 信号口径：`盘后信号 -> 次日开盘执行`
- 研究用途：
  1. 先重验 `advanced_ml`
  2. 再把 `deep_alpha` 放到同一框架下比较

### 接下来的研究顺序
1. 先用历史滚动 `liquid500` 重验 `advanced_ml`
2. 再用同一框架重验 `deep_alpha`
3. 只有在动态历史股票池 + `next_open` 下也能赢，才允许任何研究结果接近执行端

## 2026-03-20 正式框架初次结论更新
### 已完成
1. 已在历史滚动 `liquid500` + `next_open` 下，正式重验 `advanced_ml`
2. 已把 `deep_alpha` 放到同一框架、同一口径、同一 holdout 下做正面对照

### 当前结果
- `advanced_ml`
  - holdout：`2025-03-07 -> 2026-03-19`
  - 超额收益：`12.44%`
  - 超额 Sharpe：`0.323`
  - 超额回撤：`-41.39%`
- `deep_alpha`
  - 同一 holdout
  - 超额收益：`30.12%`
  - 超额 Sharpe：`1.048`
  - 超额回撤：`-17.86%`

### 这意味着什么
- 在“历史滚动高流动性股票池 + next_open”这套正式研究框架下：
  - `deep_alpha` 重新证明了自己在高流动性专用模式下的优势
  - 而且优势不只是收益，更包括更好的风险调整后收益和更浅的回撤
- 因此，`deep_alpha` 不再只是“方向上有希望”的研究分支
  - 它现在重新进入了**执行前最后验证阶段候选**

### 当前决策
- 执行端先不切换
  - 继续冻结为：`advanced_ml + liquid500 + next_open`
- 研究端主线继续保持：`deep_alpha`
- 但接下来的目标已经变化：
  - 不再先做大范围模型发散
  - 而是先做**正式框架下的多窗口复验 / walk-forward 验证**

### 接下来的最高优先级
1. 在同一正式框架下，对 `deep_alpha` 做更多 holdout / walk-forward 复验
2. 如果优势继续保持，再进入 shadow mode
3. 只有 shadow 也稳定，才讨论替代执行端默认主线

## 2026-03-20 正式框架下的多窗口 walk-forward 结论更新
### 已完成
- 已在同一正式框架下完成 `deep_alpha` vs `advanced_ml` 的多窗口 walk-forward 对照：
  - 历史滚动 `liquid500`
  - `next_open`
  - 同池同口径
- 汇总：`daily_research/output/deep_alpha_rolling_liq500_walkforward_compare_20260320.csv`

### 当前结论
- `deep_alpha` 的单窗口优势真实存在，但多窗口表现**尚不稳定**。
- 它在：
  - `2024-08-22 -> 2025-03-05`
  - `2025-09-05 -> 2026-03-19`
  表现较强；
  但在：
  - `2025-03-06 -> 2025-09-04`
  明显弱于 `advanced_ml`。
- 因此：
  - `deep_alpha` 仍然是最有前景的研究主线；
  - 但**还没有资格进入 `shadow mode`**；
  - 执行端继续冻结为：`advanced_ml + liquid500 + next_open`。

### 接下来的最高优先级
1. 先拆解 `2025-03-06 -> 2025-09-04` 这个弱窗口的失效结构。
2. 所有后续 `deep_alpha` 研究统一保留在正式框架下：
   - 历史滚动 `liquid500`
   - `next_open`
   - 多窗口 walk-forward
3. 只有当 `deep_alpha` 在多窗口上也稳定优于 `advanced_ml`，才重新讨论 `shadow mode`。

## 2026-03-21 定向 loss 放大路线的最新判断
### 已验证
- 已在正式框架下验证：
  - `trend_up_low_vol`
  - `neutral_mixed / pullback_rebound / trend_breakout`
  的定向排序增强
- 新模式：`state_targeted_rank`
- 汇总：`daily_research/output/deep_alpha_state_targeted_rank_compare_20260321.csv`

### 结论
- 这条路线当前**不成立**。
- 它在三个 walk-forward 窗口里都比 `deep_alpha shared` 更差。
- 因此：
  - 后续不再优先沿“直接放大 state-targeted ranking loss”继续深挖
  - `deep_alpha` 研究主线继续保留，但方向要收敛到：
    - 更轻的结构表达增强
    - 更稳的第一层 return 表示学习
    - 仍然固定在正式框架下验证

### 固定约束不变
1. 执行端继续冻结为：`advanced_ml + liquid500 + next_open`
2. `deep_alpha` 所有新方案继续固定在：
   - 历史滚动 `liquid500`
   - `next_open`
   - 多窗口 walk-forward
3. 只有多窗口稳定赢过 `advanced_ml`，才重新讨论 `shadow mode`

## 2026-03-21 弱窗口诊断后的研究收敛
### 已确认的问题
- `deep_alpha` 的失效不是“高流动性失效”，而是：
  - 在 `trend_up_low_vol` 这个关键状态里，排序稳定性不够
- 当前最主要的拖累结构是：
  - `neutral_mixed`
  - `pullback_rebound`
  - `trend_breakout`
- `low_vol_trend` 仍然偏弱，暂时不适合当作强化重点

### 接下来的研究方向
1. 继续增强第一层 return 学习，但只围绕：
   - `trend_up_low_vol`
   - `neutral_mixed / pullback_rebound / trend_breakout`
   这些真正影响弱窗口表现的结构
2. 对 `low_vol_trend` 先做保护或降权，不做激进强化
3. 继续保留流动性高专用研究定位，不再回到全A泛化主线

### 固定约束
1. 执行端继续冻结为：`advanced_ml + liquid500 + next_open`
2. 后续所有 `deep_alpha` 方案必须继续在：
   - 历史滚动 `liquid500`
   - `next_open`
   - 多窗口 walk-forward
   下验证
3. 没有多窗口稳定优势，不讨论 `shadow mode`

## 2026-03-21 执行与研究口径冻结
- 执行端继续冻结为：`advanced_ml + liquid500 + next_open`。
- 执行端 `liquid500` 股票池：`每日盘后更新一次`。
- 正式研究端高流动性股票池：使用**历史滚动** `liquid500/liquid800`，默认每 `21` 个交易日重建一次。
- `liquid500_latest.txt / liquid800_latest.txt` 仍然允许用于快速研究迭代，但只能作为诊断池，不作为正式结论依据。
- 从现在开始，研究和执行的口径治理规则固定为：
  1. 执行端看每日最新液态股票池；
  2. 正式研究看历史滚动股票池；
  3. 任何候选策略若要接近执行端，必须先过正式研究框架，再看是否需要进入 shadow mode。


## 2026-03-21 轻量结构表达增强复验后的约束更新
- 已验证：把 `state_id / liquidity_bucket / structure_id` 一起作为轻量上下文嵌入输入 `deep_alpha`，当前版本不成立。
- 原因不是“结构信息完全没价值”，而是当前这版上下文表达过于混合，破坏了强窗口表现。
- 从现在开始，关于结构增强的研究约束更新为：
  1. 不再直接扭 ranking loss 权重；
  2. 不再一次性把 state / liquidity / structure 三类上下文全部塞进第一层表示；
  3. 后续只尝试更轻、更单一的结构表达增强；
  4. 所有新方案继续固定在：历史滚动 `liquid500`、`next_open`、多窗口 walk-forward。

## 2026-03-22 辅助结构识别任务后的更新
- 已验证一条新的轻量结构表达路线：
  - 给 `deep_alpha` 增加辅助结构识别任务；
  - 不直接改 `ranking loss` 权重；
  - 继续固定在：历史滚动 `liquid500`、`next_open`、多窗口 walk-forward。
- 当前结果说明：
  - 结构标签是可学的；
  - 但辅助结构识别任务并没有稳定提升 return 排序；
  - 它修复了最弱窗口，却明显破坏了原本更强的窗口。
- 因此从现在开始：
  1. `aux_structure_task` 不升级为主线；
  2. 结构增强继续保留“更轻、更单一”的方向；
  3. 下一步优先尝试：
     - 结构原型；
     - 或结构平滑通道；
  4. 在多窗口没有稳定优势之前，不重谈 `shadow mode`。

## 2026-03-22 结构原型增强后的更新
- 已在正式框架下验证“结构原型增强”。
- 结论与辅助结构识别任务类似：
  - 能学到结构信息；
  - 对最弱窗口有小幅修复；
  - 但仍然破坏了强窗口。
- 因此从现在开始：
  1. `structure_prototype` 不升级为主线；
  2. 后续不再继续堆叠“能识别结构但不能稳定提升排序”的线路；
  3. 下一步优先尝试更轻的结构平滑通道，或更细的单结构局部表达；
  4. 多窗口没有稳定优势之前，仍不重谈 `shadow mode`。

## 2026-03-22 Representation-Learning Mainline
### Execution remains frozen
- Execution stays frozen at `advanced_ml + liquid500 + next_open`.
- No new research result enters execution or `shadow mode` before it passes the formal framework.

### Research mainline
- `deep_alpha` remains the only research mainline.
- The next major direction is:
  - `patch-based masked self-supervised pretraining`
  - `return ranking fine-tune`

### Fixed sequence
1. Pretrain a `patch_transformer` encoder with masked self-supervision.
2. Fine-tune the pretrained encoder for return ranking.
3. Run one formal A/B:
   - `shared baseline`
   - vs `masked-pretrained encoder + ranking fine-tune`
4. Keep all formal validation under:
   - rolling `liquid500`
   - `next_open`
   - multi-window walk-forward

### Stop rules
- Stop if the pretrained version only repairs weak windows but breaks strong windows.
- Stop if it is not more stable than `shared` across walk-forward windows.
- Do not continue polishing local gains.

## 2026-03-22 Formal verdict on masked pretraining v1
### What we tested
- `patch-based masked self-supervised pretraining`
- followed by `return ranking fine-tune`
- under the formal framework:
  - rolling `liquid500`
  - `next_open`
  - multi-window walk-forward

### Verdict
- `masked pretraining v1` does not pass.
- It fails the stop rule because:
  1. it does not improve multi-window stability
  2. it makes the weak window worse
  3. it damages the strongest window

### Current status after this verdict
- Execution remains frozen at `advanced_ml + liquid500 + next_open`
- `deep_alpha` remains the research mainline
- This exact masked-pretraining recipe is stopped; we do not keep polishing it locally

### Rule going forward
- Big-direction experiments are allowed
- but each one gets one formal verdict under the same framework
- if it fails the stop rule, we stop and move to the next idea

## 2026-03-22 Update after pretraining-epoch sensitivity
### What changed
- We tested whether the masked-pretraining route failed mainly because pretraining was too short.
- The answer is: partly yes.

### Current verdict
- `4 epoch` was too weak and gave a misleadingly bad first verdict.
- `8 epoch` is still not enough.
- `12 epoch` is the first version that:
  - repairs the weak window
  - keeps the strong window usable
- So the route is not stopped yet.

### New next step
- Run one more full formal verdict for the same recipe with `12 epoch` pretraining.
- Keep the same formal framework:
  - rolling `liquid500`
  - `next_open`
  - multi-window walk-forward
- Only after that full verdict do we decide whether the route stays alive or is stopped.

## 2026-03-22 Training-control policy
### New guardrail
- Formal deep-alpha research must now check training diagnostics before a recipe is judged.
- If a run ends with `status = undertrained`, we do **not** treat it as a final recipe verdict.

### What counts as solved
- Both masked pretraining and ranking fine-tune now use:
  - best-checkpoint restore
  - plateau LR reduction
  - early stopping after a minimum epoch floor
  - explicit diagnostics in `metrics.json`

### Local-machine runtime policy
- Current local hardware:
  - `Ryzen 7 4800H`
  - `RTX 2060 6GB`
  - Windows
- Safe defaults for deep-alpha research:
  - `patch_transformer` batch size at or below `192`
  - `num_workers <= 2`
  - `prefetch_factor = 1`
  - `AMP = on`
- We prefer safe throughput over aggressive multiprocessing, because host stability matters more than small speed gains.

## 2026-03-22 Formal runtime simplification
### Purpose
- Keep the formal research framework unchanged.
- Remove the heaviest post-training bottleneck before continuing the 12-epoch masked-pretraining verdict.

### Decision
- `score_head` and `state_gate` are now fitted on the most recent train-side window by default.
- Default window:
  - `train_eval_window_days = 126`
- This is treated as the new formal default because it is:
  - closer to recent effectiveness
  - materially lighter than full-train calibration

### Research discipline
- Do not resume full-train calibration unless there is a specific reason.
- Continue the 12-epoch masked-pretraining verdict only after this lighter formal calibration path is in place.

## 2026-03-22 Status after the lighter formal full verdict
### What is now settled
- The lighter formal path is now the official path for this research line:
  - rolling `liquid500`
  - `next_open`
  - `train_eval_window_days = 126`
  - explicit `num_workers = 0` on this machine
- The complete `12 epoch` masked-pretraining walk-forward verdict has now been re-run under that lighter formal path.

### Current result
- Window `2024-08-22 -> 2025-03-05`
  - pretrained version improves excess return / Sharpe versus `shared`
  - but drawdown is still deeper
- Window `2025-03-06 -> 2025-09-04`
  - pretrained version partially repairs the weak window
  - but does not turn it into a decisive win
- Window `2025-09-05 -> 2026-03-19`
  - pretrained version still lags the `shared` baseline on return / Sharpe
  - although drawdown is shallower

### Most important constraint
- The route still cannot receive a final verdict yet, because the pretraining diagnostics remain:
  - window `1`: `undertrained`
  - window `3`: `undertrained`
- Under the current research rule, `undertrained` means:
  - no final recipe verdict
  - no promotion toward shadow mode

### Next step
- Keep the formal framework unchanged.
- Do **not** add strict resume yet.
- Raise the masked-pretraining budget from `12 -> 16` only where it is still clearly undertrained:
  - window `1`
  - window `3`
- Re-run those windows under the same formal framework before deciding whether the recipe survives.

## 2026-03-22 Training-budget policy upgrade
### New rule
- Pretraining budget no longer has to be treated as a rigid fixed cap.
- We now support adaptive extension inside one pretraining run:
  - start from the requested epoch cap
  - if diagnostics still say `undertrained`, extend automatically
  - stop at a hard `max_total_epochs`

### Why this matters
- It reduces the chance of killing a promising recipe only because the first budget guess was too small.
- It is also cleaner than manually relaunching a series of `12 -> 16 -> 20` pretraining jobs while losing optimizer / scheduler continuity each time.

### Guardrail
- Adaptive pretraining does **not** replace formal judgment.
- A recipe still needs:
  - non-`undertrained` diagnostics
  - multi-window walk-forward
  - stable behavior across strong and weak windows


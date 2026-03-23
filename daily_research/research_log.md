# 研究日志

## 总览
- 2026-03-14：初始化 `daily_research` baseline。
- 2026-03-17：完成第一阶段研究框架扩展。
- 当前项目已经形成三层分工：
  - `baseline / advanced_ml`
  - `execution`
  - `deep_alpha`
- 当前执行端冻结为：`advanced_ml + liquid500 + next_open`
- `deep_alpha` 是研究主线，但尚未晋级执行端。

## 阅读说明
- 本文件只保留按时间顺序排列的实验记录、结果与结论。
- 当前状态与使用入口见 `README.md`。
- 当前主线、优先级与停止规则见 `daily_research_plan.md`。

## 阶段索引
- `2026-03-17 ~ 2026-03-18`
  - 基线收敛：`score + 5d`、市场状态过滤、`up_low_breakout_v2`
- `2026-03-18 ~ 2026-03-20`
  - `advanced_ml` 主线形成，并接入执行端
- `2026-03-19 ~ 2026-03-22`
  - `deep_alpha` 从烟测走向正式研究框架
- `2026-03-22`
  - 项目治理：入口去重、Git 纳管、归档规则、文档重整

## 阶段实验记录模板
- 日期：
- 实验标签：
- 数据来源：
- 股票池范围：
- 基准：
- 样本区间：
- 持仓数量：
- 权重方式：
- 调仓频率：
- 分数阈值：
- 核心指标：
- IC / RankIC 重点：
- 分层收益重点：
- 结论：
- 下一步动作：

## 2026-03-17 第一轮全A结果与实现方式选择
- 基线调优版：`all_a_stage1_tuned_concentrated_20240101`
  - 配置：`equal + 1d`
  - 结果：累计收益 `29.66%`，超额收益 `-5.32%`，超额 Sharpe `-0.154`
- 分数加权：`all_a_stage1_score_20240101`
  - 配置：`score + 1d`
  - 结果：累计收益 `33.24%`，超额收益 `-2.70%`，超额 Sharpe `-0.073`
- 低换手版本：`all_a_stage1_equal_5d_20240101`
  - 配置：`equal + 5d`
  - 结果：累计收益 `58.88%`，超额收益 `16.02%`，超额 Sharpe `0.457`
- 当时最佳配置：`all_a_stage1_score_5d_20240101`
  - 配置：`score + 5d`
  - 结果：累计收益 `89.86%`，超额收益 `38.64%`，超额 Sharpe `0.959`
- 结论：
  - `score` 优于 `equal`
  - `5d` 明显优于 `1d`
  - 研究主线切换为 `5d + score`

## 2026-03-17 因子删减与权重收敛
- 输出目录：`daily_research/output/ablation_20260317`
- 关键对比：
  - `score_5d_structure_back`：超额收益 `32.97%`，超额 Sharpe `0.842`
  - `score_5d_capped`：超额收益 `31.44%`，超额 Sharpe `0.827`
  - `score_5d_core4`：明显变弱
  - `score_5d_no_structure`：进一步变弱
- 最终保留的核心因子：
  - `volume_contraction`
  - `price_volume_divergence`
  - `atr_14_pct`
  - `volatility_20`
  - `volatility_contraction`
  - `close_strength`
- 结论：
  - 趋势组整体不适合做全局主线
  - 结构组不能完全删除

## 2026-03-17 参数扫描：holding_count 与 max_weight
- 输出目录：`daily_research/output/param_scan_20260317`
- 最优高收益配置：
  - `holding_count=5`
  - `max_weight=0.25`
  - 累计收益 `115.03%`
  - 超额收益 `57.02%`
  - 超额 Sharpe `1.222`
  - 最大回撤 `-16.72%`
- 次优但更稳健：
  - `holding_count=12`
  - `max_weight=0.25`
  - 累计收益 `100.04%`
  - 超额收益 `46.07%`
  - 超额 Sharpe `1.193`
  - 最大回撤 `-11.98%`
- 结论：
  - 高集中优于过度分散
  - `max_weight=0.25` 优于 `0.15/0.20`

## 2026-03-17 分窗口稳健性验证
- 输出目录：`daily_research/output/window_robustness_20260317`
- 配置：`holding_count=5`，`max_weight=0.25`，`score + 5d`
- 年度结果：
  - `2022`：超额收益 `-31.81%`
  - `2023`：超额收益 `-17.63%`
  - `2024`：超额收益 `65.02%`
  - `2025`：超额收益 `-15.52%`
  - `2026_ytd`：超额收益 `13.77%`
- 结论：
  - 策略存在明显市场环境依赖
  - 后续主线转向市场状态过滤

## 2026-03-17 市场状态过滤初版
- 过滤逻辑：
  - `benchmark_close > MA60`
  - `20日年化波动率 <= regime_max_annual_vol`
- `2022-2026YTD` 阈值测试：
  - `0.24`：超额收益 `37.06%`，超额 Sharpe `0.413`
  - `0.28`：超额收益 `34.77%`，超额 Sharpe `0.389`
  - `0.32`：超额收益 `45.64%`，超额 Sharpe `0.506`
- 最佳阈值：`0.32`
- 结论：
  - 市场状态过滤有效
  - 但 `2025` 仍然失效

## 2026-03-17 行业集中度约束
- 核心结论：
  - 行业约束可作为风险开关
  - 但对当前主线收益提升帮助不大
  - 不进入默认主线

## 2026-03-18 风格偏置约束
- 新增能力：
  - `load_style_map_from_tq()`
  - `--style-cap`
  - `--max-style-weight`
- `2022-2026YTD` 结果：
  - 无风格约束：超额收益 `45.64%`，超额 Sharpe `0.506`
  - `max_style_weight=0.60`：超额收益 `49.36%`，超额 Sharpe `0.545`
  - `max_style_weight=0.50`：超额收益 `49.36%`，超额 Sharpe `0.545`
- 结论：
  - 风格约束有小幅增益
  - 但不是根本解

## 2026-03-18 市场状态四象限分析

### 不加过滤的四象限归因
- 输出目录：`daily_research/output/quadrant_unfiltered_style_20260318`
- 关键发现：
  - `trend_down_low_vol` 是主要失效环境
  - 该象限样本最多，且策略几乎全程持仓
  - `trend_up_high_vol` 是最强赚钱环境

### 仅允许 `trend_up_low_vol` 的过滤结果
- 输出目录：`daily_research/output/quadrant_filtered_style_20260318`
- 整体结果：
  - 累计收益 `+32.36%`
  - 超额收益 `+40.88%`
  - 超额 Sharpe `0.425`
- 结论：
  - 切掉了最差环境
  - 但也错杀了 `trend_up_high_vol`

## 2026-03-18 市场状态白名单升级
- 白名单升级为：
  - `trend_up_low_vol`
  - `trend_up_high_vol`
- 对比结果：
  - 旧过滤：累计收益 `+32.36%`，超额收益 `+40.88%`
  - 新过滤：累计收益 `+81.63%`，超额收益 `+93.14%`
- 结论：
  - 之前的问题不是过滤方向错
  - 而是过滤过粗

## 2026-03-18 2025 细分诊断：`trend_up_low_vol`
- 输出目录：`daily_research/output/quadrant_factor_diag_up_low_20260318`
- 发现：
  - 综合分数在 `2025 / trend_up_low_vol` 中并未失效
  - 但超额转换很弱
  - 更强的单因子包括：
    - `breakout_20`
    - `drawdown_20`
    - `range_position_20`
    - `volatility_20`
    - `up_day_ratio_10`
- 结论：
  - 问题不是没有 alpha
  - 而是上涨低波环境下组合过于防守

## 2026-03-18 状态内动态权重实验：`up_low_breakout_v1`
- 输出目录：
  - 基线：`daily_research/output/state_alpha_baseline_20220101`
  - 动态权重：`daily_research/output/state_alpha_up_low_breakout_20220101`
- 结果：
  - 基线：累计收益 `+11.32%`，超额收益 `+18.39%`，超额 Sharpe `0.232`
  - `up_low_breakout_v1`：累计收益 `+154.97%`，超额收益 `+171.16%`，超额 Sharpe `1.392`
- 关键改善：
  - `2025` 超额从 `-21.86%` 提升到 `+30.64%`
  - `trend_up_low_vol` 超额从 `-44.63%` 提升到 `+20.45%`
- 结论：
  - 状态内动态权重有效

## 2026-03-18 状态内动态权重实验二：`up_dual_v1`
- 输出目录：`daily_research/output/state_alpha_up_dual_20220101`
- 结果：
  - `up_low_breakout_v1`：累计收益 `+154.97%`，超额收益 `+171.16%`
  - `up_dual_v1`：累计收益 `+179.12%`，超额收益 `+197.58%`
- 关键改善：
  - `trend_up_high_vol` 超额从 `+9.59%` 提升到 `+23.50%`
- 结论：
  - `up_dual_v1` 优于 `up_low_breakout_v1`
  - 当前是更强的候选主线

## 2026-03-18 滚动样本外验证：状态 Profile 选择
- 验证脚本：`daily_research/baseline/walkforward_profile_validation.py`
- 输出目录：`daily_research/output/walkforward_profiles_2y_20260318`
- 方法：
  - 每年只用前 2 年训练窗口
  - 按 `excess_sharpe` 选择 profile
  - 在下一年测试
- 候选 profile：
  - `none`
  - `up_low_breakout_v1`
  - `up_dual_v1`
- 整体结果：
  - 累计收益 `+87.35%`
  - 超额收益 `+53.87%`
  - 超额 Sharpe `0.843`
- 每年被选中的 profile：
  - 2023：`up_low_breakout_v1`
  - 2024：`none`
  - 2025：`none`
  - 2026：`none`
- 结论：
  - `up_dual_v1` 在全样本中很强
  - 但用“过去两年总体 Sharpe”做事前选择时，不能稳定被选出来

## 2026-03-18 冻结时点验证：2021-2023 设计，2024-2026 测试
- 验证脚本：`daily_research/baseline/frozen_profile_validation.py`
- 输出目录：`daily_research/output/frozen_profiles_20210101_20231231_to_20260318`
- 方法：
  - 训练期固定为 `2021-2023`
  - 只在训练期比较候选 profile
  - 测试期固定为 `2024-2026`
  - 中途不滚动改规则
- 候选 profile：
  - `none`
  - `up_low_breakout_v1`
  - `up_dual_v1`
- 结论：
  - `up_dual_v1` 的全样本优势不够稳健
  - `none` 是当前更稳的样本外基线
  - `up_low_breakout_v1` 更偏收益进攻型备选

## 2026-03-18 基于象限内 RankIC 的 profile 激活验证
- 验证脚本：`daily_research/baseline/quadrant_ic_activation_validation.py`
- 输出目录：
  - `daily_research/output/quadrant_ic_activation_2y_20260318`（`10d RankIC`）
  - `daily_research/output/quadrant_ic_activation_2y_h20_20260318`（`20d RankIC`）
- 方法：
  - 每年用前 2 年训练窗口
  - 不再按总体 Sharpe 选 profile
  - 改为按每个象限里的 `RankIC` 选择下一年启用哪套 profile
- 结论：
  - 无论 `10d` 还是 `20d`，结果仍然全部偏向 `none`
  - 说明当前动态 profile 的事前横截面优势还不够稳定

## 2026-03-18 冻结时点下的 `trend_up_low_vol` 因子重设：`up_low_breakout_v2`
- 目标：
  - 在不破坏 `none` 样本外稳健性的前提下，重做 `trend_up_low_vol` 的状态内权重
- 思路：
  - 保留更多 `none` 的低波/量价骨架
  - 只适度加入突破与结构因子
  - 避免 `v1` 那种过强的进攻化改动
- 新 profile：`up_low_breakout_v2`
- 输出目录：`daily_research/output/frozen_up_low_redesign_20260318`
- 候选：
  - `none`
  - `up_low_breakout_v1`
  - `up_low_breakout_v2`

### 训练期（2021-2023）
- `none`
  - 超额收益 `21.84%`
  - 超额 Sharpe `0.481`
- `up_low_breakout_v2`
  - 超额收益 `20.24%`
  - 超额 Sharpe `0.448`
- `up_low_breakout_v1`
  - 超额收益 `16.99%`
  - 超额 Sharpe `0.367`

### 测试期（2024-2026）
- `none`
  - 累计收益 `114.35%`
  - 超额收益 `56.38%`
  - 超额 Sharpe `1.218`
- `up_low_breakout_v1`
  - 累计收益 `117.37%`
  - 超额收益 `58.59%`
  - 超额 Sharpe `1.063`
- `up_low_breakout_v2`
  - 累计收益 `124.89%`
  - 超额收益 `64.07%`
  - 超额 Sharpe `1.291`
  - 超额最大回撤 `-16.56%`

### 结论
- `up_low_breakout_v2` 明显优于 `up_low_breakout_v1`
- 与 `none` 相比，`v2` 在训练期仍略弱，但差距已经明显缩小
- 更重要的是，`v2` 在测试期同时实现了：
  - 更高的超额收益
  - 更高的超额 Sharpe
  - 更低的超额回撤
- 这说明 `trend_up_low_vol` 的重设方向是有效的，而且比之前更接近“冻结时点可接受”的候选版本

## 2026-03-18 `none` 与 `up_low_breakout_v2` 的双模型激活验证
- 验证脚本：`daily_research/baseline/quadrant_ic_activation_validation.py`
- 输出目录：
  - `daily_research/output/quadrant_ic_activation_none_vs_v2_h10_20260318`
  - `daily_research/output/quadrant_ic_activation_none_vs_v2_h20_20260318`
- 方法：
  - 只比较两套模型：
    - `none`
    - `up_low_breakout_v2`
  - 仍然使用前 2 年训练窗口
  - 仍然按象限内 `RankIC` 决定下一年启用哪套

### 10d RankIC 结果
- 整体结果：
  - 累计收益 `+104.26%`
  - 超额收益 `+66.64%`
  - 超额 Sharpe `0.997`
- 激活情况：
  - 所有年份、所有象限仍然都选中 `none`
- 结论：
  - `10d RankIC` 下，`v2` 虽然比旧动态版本更接近 `none`，但仍不足以稳定胜出

### 20d RankIC 结果
- 整体结果：
  - 累计收益 `+115.93%`
  - 超额收益 `+76.06%`
  - 超额 Sharpe `1.107`
- 激活情况：
  - `2023`：全部 `none`
  - `2024`：全部 `none`
  - `2025`：全部 `none`
  - `2026`：
    - `trend_up_low_vol` 选中 `up_low_breakout_v2`
    - `trend_up_high_vol` 仍为 `none`
- 结论：
  - 这是目前第一次出现增强版在“事前激活”里胜过 `none`
  - 说明 `up_low_breakout_v2` 比之前的动态版本更接近可被稳定激活的状态

### 当前判断
- `none` 仍是主基线。
- `up_low_breakout_v2` 是当前唯一一个在冻结验证和激活验证中都出现正面信号的增强版本。
- 下一步最值得做的是：
  1. 固定高波部分继续使用 `none`
  2. 只围绕 `trend_up_low_vol` 的 `v2` 做进一步激活优化

## 2026-03-18 `up_dual_v2` 冻结验证
- 目标：
  - 用 `up_low_breakout_v2` 替换 `up_dual_v1` 的低波部分，测试双状态版本能否一起受益
- 新 profile：`up_dual_v2`
  - `trend_up_low_vol`：`up_low_breakout_v2`
  - `trend_up_high_vol`：沿用高波动量增强
- 输出目录：`daily_research/output/frozen_up_dual_v2_20260318`
- 对比对象：
  - `none`
  - `up_low_breakout_v2`
  - `up_dual_v2`

### 训练期（2021-2023）
- `none`
  - 超额收益 `21.84%`
  - 超额 Sharpe `0.481`
- `up_low_breakout_v2`
  - 超额收益 `20.24%`
  - 超额 Sharpe `0.448`
- `up_dual_v2`
  - 超额收益 `20.24%`
  - 超额 Sharpe `0.448`

### 测试期（2024-2026）
- `none`
  - 累计收益 `114.35%`
  - 超额收益 `56.08%`
  - 超额 Sharpe `1.212`
- `up_low_breakout_v2`
  - 累计收益 `124.89%`
  - 超额收益 `63.75%`
  - 超额 Sharpe `1.285`
  - 超额最大回撤 `-16.56%`
- `up_dual_v2`
  - 累计收益 `84.04%`
  - 超额收益 `34.01%`
  - 超额 Sharpe `0.655`
  - 超额最大回撤 `-22.23%`

### 结论
- `up_dual_v2` 依然明显弱于 `up_low_breakout_v2`。
- 这说明当前的高波上涨增强层仍然不稳健，是双状态版本的主要拖累。
- 到目前为止，更合理的研究结论是：
  - `none`：稳健基线
  - `up_low_breakout_v2`：当前最有希望的增强版本
  - `up_dual_v1 / up_dual_v2`：暂不进入主线，先降级为实验分支

## 2026-03-18 `none` 与 `up_low_breakout_v2` 的季度激活验证
- 目标：
  - 不再继续扩新 profile，而是把研究重心转到更可执行的事前激活机制上。
  - 固定 `trend_up_high_vol` 继续使用 `none`，只在 `trend_up_low_vol` 内部比较：
    - `none`
    - `up_low_breakout_v2`
- 代码更新：
  - `daily_research/baseline/quadrant_ic_activation_validation.py`
  - 新增：
    - `--activation-frequency`
    - `--quadrant-candidates`
- 关键命令：
  - 半年度激活：
    - `python daily_research/baseline/quadrant_ic_activation_validation.py --start-date 20210101 --benchmark 000300.SH --train-years 2 --rebalance-freq 5d --regime-max-annual-vol 0.32 --regime-quadrants trend_up_low_vol,trend_up_high_vol --style-cap --max-style-weight 0.50 --profiles none,up_low_breakout_v2 --quadrant-candidates "trend_up_low_vol=none,up_low_breakout_v2;trend_up_high_vol=none" --ic-horizon 20 --activation-frequency halfyear --experiment-tag quadrant_ic_activation_none_v2_halfyear_h20_20260318`
  - 季度激活：
    - `python daily_research/baseline/quadrant_ic_activation_validation.py --start-date 20210101 --benchmark 000300.SH --train-years 2 --rebalance-freq 5d --regime-max-annual-vol 0.32 --regime-quadrants trend_up_low_vol,trend_up_high_vol --style-cap --max-style-weight 0.50 --profiles none,up_low_breakout_v2 --quadrant-candidates "trend_up_low_vol=none,up_low_breakout_v2;trend_up_high_vol=none" --ic-horizon 20 --activation-frequency quarter --experiment-tag quadrant_ic_activation_none_v2_quarter_h20_20260318`
  - 同跨度固定对照：
    - `python daily_research/baseline/frozen_profile_validation.py --start-date 20210101 --train-end 20220331 --test-start 20220401 --benchmark 000300.SH --rebalance-freq 5d --regime-max-annual-vol 0.32 --regime-quadrants trend_up_low_vol,trend_up_high_vol --style-cap --max-style-weight 0.50 --profiles none,up_low_breakout_v2 --experiment-tag frozen_none_v2_from_20220401_20260318`
- 输出目录：
  - `daily_research/output/quadrant_ic_activation_none_v2_halfyear_h20_20260318`
  - `daily_research/output/quadrant_ic_activation_none_v2_quarter_h20_20260318`
  - `daily_research/output/frozen_none_v2_from_20220401_20260318`

### 同类方案对比
- 年度 `20d RankIC` 激活：
  - 累计收益 `115.93%`
  - 超额收益 `76.06%`
  - 超额 Sharpe `1.107`
  - 超额最大回撤 `-16.98%`
- 半年度 `20d RankIC` 激活：
  - 累计收益 `94.07%`
  - 超额收益 `85.96%`
  - 超额 Sharpe `1.020`
  - 超额最大回撤 `-21.82%`
- 季度 `20d RankIC` 激活：
  - 累计收益 `119.59%`
  - 超额收益 `116.29%`
  - 超额 Sharpe `1.194`
  - 超额最大回撤 `-19.86%`

### 同跨度固定对照（2022-04-01 至今）
- `none`
  - 累计收益 `95.35%`
  - 超额收益 `79.32%`
  - 超额 Sharpe `0.876`
  - 超额最大回撤 `-20.96%`
- `up_low_breakout_v2`
  - 累计收益 `103.23%`
  - 超额收益 `86.55%`
  - 超额 Sharpe `0.918`
  - 超额最大回撤 `-27.28%`

### 季度激活的实际选择
- `trend_up_high_vol`
  - 所有季度都固定为 `none`
- `trend_up_low_vol`
  - `2022Q3`：`up_low_breakout_v2`
  - `2025Q4`：`up_low_breakout_v2`
  - `2026Q1`：`up_low_breakout_v2`
  - 其余季度：`none`

### 结论
- 目前最有前景的事前激活方案，不是继续扩新 profile，而是：
  - `trend_up_high_vol` 固定使用 `none`
  - `trend_up_low_vol` 在 `none` 与 `up_low_breakout_v2` 之间做 `20d RankIC` 的季度切换
- 这套季度激活方案已经优于：
  - 年度激活
  - 半年度激活
  - 同跨度固定 `none`
  - 同跨度固定 `up_low_breakout_v2`
- 这说明我们已经从“增强版本是否有效”推进到了“增强版本何时应该被打开”这个更有研究价值的阶段。
- 下一步最值得做的是：
  1. 检查季度激活是否能进一步简化成更可解释的规则
  2. 对训练窗口长度做稳健性验证，例如 `1.5 / 2 / 3` 年

## 2026-03-18 季度激活的训练窗口长度稳健性验证
- 目标：
  - 检查当前最优的季度激活方案，是否对训练窗口长度敏感。
  - 继续固定：
    - `trend_up_high_vol`：`none`
    - `trend_up_low_vol`：在 `none / up_low_breakout_v2` 间做 `20d RankIC` 激活
- 代码更新：
  - `daily_research/baseline/quadrant_ic_activation_validation.py`
  - 新增 `--train-months`，允许直接指定训练窗口月份数
- 关键命令：
  - `18` 个月：
    - `python daily_research/baseline/quadrant_ic_activation_validation.py --start-date 20210101 --benchmark 000300.SH --train-years 2 --train-months 18 --rebalance-freq 5d --regime-max-annual-vol 0.32 --regime-quadrants trend_up_low_vol,trend_up_high_vol --style-cap --max-style-weight 0.50 --profiles none,up_low_breakout_v2 --quadrant-candidates "trend_up_low_vol=none,up_low_breakout_v2;trend_up_high_vol=none" --ic-horizon 20 --activation-frequency quarter --experiment-tag quadrant_ic_activation_none_v2_quarter_h20_m18_20260318`
  - `24` 个月：
    - `python daily_research/baseline/quadrant_ic_activation_validation.py --start-date 20210101 --benchmark 000300.SH --train-years 2 --train-months 24 --rebalance-freq 5d --regime-max-annual-vol 0.32 --regime-quadrants trend_up_low_vol,trend_up_high_vol --style-cap --max-style-weight 0.50 --profiles none,up_low_breakout_v2 --quadrant-candidates "trend_up_low_vol=none,up_low_breakout_v2;trend_up_high_vol=none" --ic-horizon 20 --activation-frequency quarter --experiment-tag quadrant_ic_activation_none_v2_quarter_h20_m24_20260318`
  - `36` 个月：
    - `python daily_research/baseline/quadrant_ic_activation_validation.py --start-date 20210101 --benchmark 000300.SH --train-years 2 --train-months 36 --rebalance-freq 5d --regime-max-annual-vol 0.32 --regime-quadrants trend_up_low_vol,trend_up_high_vol --style-cap --max-style-weight 0.50 --profiles none,up_low_breakout_v2 --quadrant-candidates "trend_up_low_vol=none,up_low_breakout_v2;trend_up_high_vol=none" --ic-horizon 20 --activation-frequency quarter --experiment-tag quadrant_ic_activation_none_v2_quarter_h20_m36_20260318`
- 输出目录：
  - `daily_research/output/quadrant_ic_activation_none_v2_quarter_h20_m18_20260318`
  - `daily_research/output/quadrant_ic_activation_none_v2_quarter_h20_m24_20260318`
  - `daily_research/output/quadrant_ic_activation_none_v2_quarter_h20_m36_20260318`

### 原始结果
- `18` 个月：
  - 累计收益 `103.52%`
  - 超额收益 `110.28%`
  - 超额 Sharpe `1.248`
  - 超额最大回撤 `-19.86%`
- `24` 个月：
  - 累计收益 `122.45%`
  - 超额收益 `103.71%`
  - 超额 Sharpe `1.436`
  - 超额最大回撤 `-14.89%`
- `36` 个月：
  - 累计收益 `101.69%`
  - 超额收益 `59.53%`
  - 超额 Sharpe `1.295`
  - 超额最大回撤 `-15.86%`

### 共同测试区间对比（2024-01-02 至今）
- `18` 个月：
  - 累计收益 `134.81%`
  - 超额收益 `85.73%`
  - 超额 Sharpe `1.768`
  - 超额最大回撤 `-14.89%`
- `24` 个月：
  - 累计收益 `134.81%`
  - 超额收益 `85.73%`
  - 超额 Sharpe `1.768`
  - 超额最大回撤 `-14.89%`
- `36` 个月：
  - 累计收益 `101.69%`
  - 超额收益 `59.53%`
  - 超额 Sharpe `1.291`
  - 超额最大回撤 `-15.86%`

### 与固定版本的同跨度对照（2024-2026）
- 固定 `none`
  - 累计收益 `114.35%`
  - 超额收益 `56.38%`
  - 超额 Sharpe `1.218`
  - 超额最大回撤 `-18.09%`
- 固定 `up_low_breakout_v2`
  - 累计收益 `124.89%`
  - 超额收益 `64.07%`
  - 超额 Sharpe `1.291`
  - 超额最大回撤 `-16.56%`
- 季度激活（`24` 个月训练窗）
  - 累计收益 `134.81%`
  - 超额收益 `85.73%`
  - 超额 Sharpe `1.768`
  - 超额最大回撤 `-14.89%`

### 结论
- 对当前这套季度激活方案来说：
  - `18` 个月与 `24` 个月在共同测试区间表现一致
  - `36` 个月明显变钝，说明训练窗过长会稀释状态切换的敏感性
- 当前更合理的默认选择是：
  - `20d RankIC`
  - 季度刷新
  - `trend_up_high_vol` 固定 `none`
  - `trend_up_low_vol` 在 `none / up_low_breakout_v2` 之间切换
  - 训练窗口优先取 `24` 个月
- 到这里为止，研究重点已经从“继续发明新 profile”转向“把季度激活机制做得更可解释、更可落地”。

## 2026-03-18 季度激活规则解释化
- 目标：
  - 把当前最优的季度激活方案，从“RankIC 黑盒切换”进一步收敛成更直观、可解释的规则。
- 新增脚本：
  - `daily_research/baseline/analyze_activation_rule_candidates.py`
  - `daily_research/baseline/rule_based_activation_validation.py`
- 分析设定：
  - 焦点象限：`trend_up_low_vol`
  - 高波上涨象限继续固定 `none`
  - 训练窗口：`24` 个月
  - 刷新频率：季度
  - 评估口径：`20d RankIC`
- 输出目录：
  - `daily_research/output/activation_rule_candidates_v2_q24_20260318`
  - `daily_research/output/rule_activation_none_v2_q24_20260318`

### 解释分析结果
- 在 `24` 个月训练窗下，`up_low_breakout_v2` 被激活的季度只有：
  - `2025Q4`
  - `2026Q1`
- 这两个季度共同特征很清楚：
  - `breakout_20` 的训练期 `RankIC` 转正
  - `drawdown_20` 的训练期 `RankIC` 转正
  - `range_position_20` 的训练期 `RankIC` 虽仍为负，但已经回升到 `-0.06` 以上

### 提炼出的可解释规则候选
- 若在 `trend_up_low_vol` 的过去 `24` 个月训练窗中，同时满足：
  - `breakout_20 RankIC >= 0`
  - `drawdown_20 RankIC >= 0`
  - `range_position_20 RankIC >= -0.06`
- 则下一季度启用 `up_low_breakout_v2`
- 否则继续使用 `none`

### 规则版连续回测结果
- 说明：
  - 这一轮规则验证使用的是“连续回测”口径
  - 因而与前面的“按窗口拼接”验证不是完全同一统计口径
- 结果：
  - 累计收益 `135.70%`
  - 超额收益 `96.72%`
  - 超额 Sharpe `1.340`
  - 超额最大回撤 `-12.34%`
- 激活季度与前面的季度 RankIC 版保持一致：
  - `2025Q4`
  - `2026Q1`

### 结论
- 现在我们已经有了一个“不是纯黑盒”的激活候选规则。
- 它的价值不只是结果好，而是：
  - 和 `RankIC` 版激活的关键季度一致
  - 规则含义能被清楚解释
  - 后续可以继续做成更稳健的研究对象
- 下一步最值得做的是：
  1. 用连续口径重做一次 `RankIC` 激活对照，统一比较基准
  2. 在规则版上做阈值稳健性验证，例如 `range_position_20` 的阈值从 `-0.07 / -0.06 / -0.05` 比较

## 2026-03-18 交付导向升级：先进版主程序
- 背景：
  - 为了尽快交付一套更强、更完整、可直接运行的程序，不再只沿着手工规则与 profile 验证推进。
  - 在现有数据层、因子层、组合层、回测层基础上，直接叠加机器学习横截面排序。
- 新增文件：
  - `daily_research/baseline/ml_alpha.py`
  - `daily_research/baseline/run_advanced_daily_research.py`
- 设计思路：
  - 延续当前有效框架：
    - 市场状态过滤
    - `none`
    - `up_low_breakout_v2`
  - 在此基础上新增：
    - `sklearn` 梯度提升树滚动训练
    - 预测未来 `20d` 超额收益
    - 将机器学习分数与 `none / v2` 基线分数做集成
- 默认程序特征：
  - 全A
  - 基准 `000300.SH`
  - 高集中组合
  - `5d` 调仓
  - 风格约束
  - 滚动重训
  - 完整输出 `equity_curve / actions / metrics / factor_ic / latest_scores / training_log`
- 烟测命令：
  - `python daily_research/baseline/run_advanced_daily_research.py --data-source tq --stocks 600000.SH,600036.SH,601318.SH,000001.SZ,000333.SZ,002415.SZ --start-date 20240101 --benchmark 000300.SH --holding-count 3 --rebalance-freq 5d --ml-train-window-days 120 --ml-min-train-dates 40 --ml-retrain-every-days 10 --ml-max-samples-per-day 100 --ml-max-train-rows 10000 --experiment-tag advanced_ml_smoke_20260318`
- 烟测输出目录：
  - `daily_research/output/advanced_ml_smoke_20260318`

### 烟测结果
- 累计收益 `30.92%`
- 年化收益 `13.64%`
- Sharpe `1.434`
- 最大回撤 `-6.72%`
- 超额收益 `-4.83%`

### 结论
- 新主程序已经跑通，不是概念代码。
- 这条线的意义不是替代所有研究，而是把已有研究成果快速压缩成一套更强的可执行程序。
- 下一步应优先在这条先进版主程序上做真实全A实验，而不是继续停留在纯规则层面。

## 2026-03-18 尾盘手工执行模式
- 目标：
  - 让程序不只输出研究结果，而是在每日尾盘前直接生成一份可执行的操作建议文本。
- 新增文件：
  - `daily_research/baseline/generate_daily_trade_plan.py`
  - `daily_research/execution/current_positions.example.csv`
- 功能：
  - 读取当前持仓与现金
  - 运行先进版主线分数
  - 自动生成：
    - 卖出 / 减仓
    - 买入 / 加仓
    - 当前持仓概览
    - 候选观察名单
  - 每次覆盖刷新：
    - `daily_research/execution/output/latest_trade_plan.txt`
- 烟测输出目录：
  - `daily_research/execution/output/latest_trade_plan.txt`

### 烟测结论
- 脚本已能成功读取持仓、计算当日目标组合，并输出中文 TXT 建议。
- 在禁止开仓的市场状态下：
  - 会自动给出卖出建议
  - 不会给出新的买入动作
  - 但仍会保留观察名单，方便次日继续跟踪

## 2026-03-18 训练与执行拆分
- 调整目标：
  - 将“模型训练”和“尾盘执行”从同一个脚本中拆开
  - 避免尾盘运行时再做整套滚动训练
- 新增文件：
  - `daily_research/baseline/train_trade_model.py`
  - `daily_research/execution/update_model.py`
- 执行端改动：
  - `daily_research/execution/run_trade_plan.py` 默认读取：
    - `daily_research/execution/models/latest_ml_model.joblib`
  - `daily_research/baseline/generate_daily_trade_plan.py` 默认不再现场训练
  - 若模型产物不存在，会明确提示先运行：
    - `daily_research/execution/update_model.py`
- 当前建议流程：
  1. 先单独更新模型产物
  2. 再在尾盘运行执行端生成建议文本
- 这样做的收益：
  - 尾盘执行速度更稳定
  - 训练与使用职责更清楚
  - 每天建议对应哪一次训练结果更容易追踪

## 2026-03-18 高杠杆升级：多周期 ML 集成
- 升级目标：
  - 将原本单一 `20d` 目标的 ML，升级为多周期集成
  - 让模型同时吸收 `5d / 10d / 20d` 三个周期的横截面信息
- 核心改动：
  - `daily_research/baseline/ml_alpha.py`
    - 新增多周期训练、预测、模型产物保存/加载支持
  - `daily_research/baseline/train_trade_model.py`
    - 支持训练多周期点时模型产物
  - `daily_research/baseline/generate_daily_trade_plan.py`
    - 默认读取离线多周期模型产物生成尾盘建议
  - `daily_research/baseline/run_advanced_daily_research.py`
    - 接入多周期 ML 集成研究主线
- 默认新方向：
  - `ml-target-horizons = 5,10,20`
- 已验证：
  - `update_model.py` 小样本训练通过
  - `run_trade_plan.py` 小样本读取模型并生成建议通过
  - `compileall` 与 `--help` 通过
- 当前判断：
  - 执行端已经能稳定使用多周期模型产物
  - 研究主线也已接入多周期能力
  - 但完整 `run_advanced_daily_research.py` 的多周期全流程实跑仍需后续继续压测与优化速度

## 2026-03-18 多周期 ML 集成首轮 A/B 对比
- 对比目的：
  - 验证多周期集成是否真的优于原来的单周期 `20d`
- 实验样本：
  - `20` 只代表性股票
  - 区间：`2024-01-01` 起
  - 组合：`holding_count=5`
  - 调仓：`5d`
  - 训练窗：`252` 个交易日
- 输出目录：
  - `daily_research/output/advanced_ml_ab_single20_20260318`
  - `daily_research/output/advanced_ml_ab_multi_5_10_20_20260318`
  - 汇总表：`daily_research/output/advanced_ml_ab_compare_20260318.csv`

### 结果
- 单周期 `20d`
  - 累计收益 `29.35%`
  - Sharpe `0.944`
  - 超额收益 `-5.97%`
  - 超额 Sharpe `-0.178`
- 多周期 `5/10/20`
  - 累计收益 `27.79%`
  - Sharpe `0.896`
  - 超额收益 `-7.11%`
  - 超额 Sharpe `-0.215`

### 初步结论
- 多周期集成不是自动更优。
- 在这轮样本里：
  - 多周期版本回撤略浅
  - 命中率略高
  - 但收益、Sharpe、超额都弱于单周期 `20d`
- 当前更合理的判断：
  - 多周期能力值得保留
  - 但不能直接把 `5/10/20` 当成默认最优组合
  - 下一步应优先研究：
    1. 不同周期的权重分配
    2. 按市场状态只启用部分周期

## 2026-03-18 多周期权重扫描
- 目标：
  - 检验“`20d` 为主、`5d/10d` 为辅”是否比单周期和简单等权更有效
- 对比组：
  - `single20`
  - `multi_equal`（等权 `5/10/20`）
  - `multi_w235`（`5:0.2,10:0.3,20:0.5`）
  - `multi_w127`（`5:0.1,10:0.2,20:0.7`）
- 输出：
  - `daily_research/output/advanced_ml_ab_weight_scan_20260318.csv`

### 结果
- `single20`
  - 累计收益 `29.35%`
  - Sharpe `0.944`
  - 超额收益 `-5.97%`
- `multi_equal`
  - 累计收益 `27.79%`
  - Sharpe `0.896`
  - 超额收益 `-7.11%`
- `multi_w235`
  - 累计收益 `31.06%`
  - Sharpe `0.982`
  - 超额收益 `-4.73%`
- `multi_w127`
  - 累计收益 `18.77%`
  - Sharpe `0.620`
  - 超额收益 `-13.66%`

### 结论
- “`20d` 主导 + `5d/10d` 辅助”是有效方向，但不能过度压缩到只剩长周期。
- 当前这轮样本里最优的是：
  - `5:0.2,10:0.3,20:0.5`
- 这组已经同时优于：
  - 单周期 `20d`
  - 多周期等权
  - 过度偏向 `20d` 的 `0.1/0.2/0.7`
- 因此当前 advanced ML 主线的多周期默认建议可先定为：
  - `--ml-target-horizons 5,10,20`
  - `--ml-horizon-weights 5:0.2,10:0.3,20:0.5`

## 2026-03-18 状态内周期权重验证
- 目标：
  - 检验“不同市场状态使用不同 ML 周期权重”是否能进一步提升多周期 advanced ML 主线
- 代码进展：
  - `ml_alpha.py` 已支持按 `quadrant` 解析状态内周期权重
  - `run_advanced_daily_research.py`
  - `train_trade_model.py`
  - `generate_daily_trade_plan.py`
  - 均已支持参数：
    - `--ml-state-horizon-profiles`
- 典型参数格式：
  - `trend_up_low_vol=5:0.15,10:0.25,20:0.60;trend_up_high_vol=5:0.30,10:0.35,20:0.35`
- 输出目录：
  - `daily_research/output/advanced_ml_ab_stateweights_a_20260318`
  - `daily_research/output/advanced_ml_ab_stateweights_b_20260318`

### 结果
- `stateweights_a`
  - `trend_up_low_vol=5:0.15,10:0.25,20:0.60`
  - `trend_up_high_vol=5:0.35,10:0.35,20:0.30`
  - 累计收益 `26.67%`
  - Sharpe `0.859`
  - 超额收益 `-7.92%`
- `stateweights_b`
  - `trend_up_low_vol=5:0.20,10:0.30,20:0.50`
  - `trend_up_high_vol=5:0.30,10:0.35,20:0.35`
  - 累计收益 `31.06%`
  - Sharpe `0.982`
  - 超额收益 `-4.73%`

### 结论
- 状态内周期权重能力已经打通，训练端和执行端都可使用。
- 但当前两轮实验里：
  - `stateweights_a` 明显弱于固定全局权重
  - `stateweights_b` 仅与固定全局权重 `5:0.2,10:0.3,20:0.5` 基本持平
- 因此截至目前，advanced ML 主线仍维持：
  - 固定多周期权重 `5:0.2,10:0.3,20:0.5`
- 状态内周期权重保留为研究能力，不进入当前默认执行配置。

## 2026-03-19 Advanced ML 集成权重扫描
- 目标：
  - 检验 `ML / none / v2` 三路分数的集成比例，确认是否存在比当前 `0.7 / 0.2 / 0.1` 更强的组合方式
- 样本设置：
  - `20` 只代表性股票
  - 区间：`2024-01-01` 起
  - 组合：`holding_count=5`
  - 调仓：`5d`
  - 多周期：`5,10,20`
  - 周期权重：`5:0.2,10:0.3,20:0.5`
- 输出：
  - `daily_research/output/advanced_ml_ensemble_scan_20260318.csv`
  - `daily_research/output/advanced_ml_ens_base_20260318`
  - `daily_research/output/advanced_ml_ens_ml85_20260318`
  - `daily_research/output/advanced_ml_ens_bal60_20260318`
  - `daily_research/output/advanced_ml_ens_rule50_20260318`
  - `daily_research/output/advanced_ml_ens_v2tilt_20260318`

### 对比组
- `base`
  - `ML=0.70, none=0.20, v2=0.10`
- `ml85`
  - `ML=0.85, none=0.10, v2=0.05`
- `bal60`
  - `ML=0.60, none=0.25, v2=0.15`
- `rule50`
  - `ML=0.50, none=0.30, v2=0.20`
- `v2tilt`
  - `ML=0.55, none=0.20, v2=0.25`

### 结果
- `base`
  - 累计收益 `31.06%`
  - Sharpe `0.982`
  - 超额收益 `-4.73%`
  - 超额 Sharpe `-0.140`
- `ml85`
  - 累计收益 `35.11%`
  - Sharpe `1.103`
  - 超额收益 `-1.78%`
  - 超额 Sharpe `-0.052`
- `bal60`
  - 累计收益 `30.75%`
  - Sharpe `0.985`
  - 超额收益 `-4.96%`
  - 超额 Sharpe `-0.146`
- `rule50`
  - 累计收益 `37.62%`
  - Sharpe `1.189`
  - 超额收益 `0.04%`
  - 超额 Sharpe `0.001`
- `v2tilt`
  - 累计收益 `35.13%`
  - Sharpe `1.101`
  - 超额收益 `-1.77%`
  - 超额 Sharpe `-0.052`

### 结论
- 这轮样本里，最优组合不是“继续提高 ML 占比”，而是：
  - `ML=0.50, none=0.30, v2=0.20`
- 这说明当前 advanced ML 更像“强增强器”，而不是“应该单独压倒规则分数”的主导源。
- 与当前默认 `0.70 / 0.20 / 0.10` 相比：
  - `rule50` 的累计收益更高
  - Sharpe 更高
  - 超额也从负值抬到了接近持平
- 但这仍然是 `20` 只股票样本的研究结果，暂不直接改成执行端默认值。
- 下一步优先级：
  1. 用更大股票池或全A再验证一次 `rule50`
  2. 如果结果保持，再考虑把执行端默认集成权重从 `0.70 / 0.20 / 0.10` 切到 `0.50 / 0.30 / 0.20`

## 2026-03-19 Advanced ML 集成权重大样本验证
- 目标：
  - 用更大样本验证上一轮 `20` 股实验里表现最好的 `rule50`
  - 判断是否应该把执行端默认集成权重从 `0.70 / 0.20 / 0.10` 切换到 `0.50 / 0.30 / 0.20`
- 样本设置：
  - 股票池：全A
  - 起始日期：`2022-01-01`
  - 股票数：`5498`
  - 日期数：`1017`
  - 组合：`holding_count=5`
  - 调仓：`5d`
  - 多周期：`5,10,20`
  - 周期权重：`5:0.2,10:0.3,20:0.5`
- 输出：
  - `daily_research/output/advanced_ml_alla_base_20220101_20260319`
  - `daily_research/output/advanced_ml_alla_rule50_20220101_20260319`
  - 汇总表：`daily_research/output/advanced_ml_ensemble_alla_compare_20260319.csv`

### 结果
- `base`
  - 权重：`ML=0.70, none=0.20, v2=0.10`
  - 累计收益 `538.81%`
  - 年化收益 `58.40%`
  - Sharpe `1.945`
  - 最大回撤 `-27.36%`
  - 超额收益 `574.39%`
  - 超额 Sharpe `2.054`
- `rule50`
  - 权重：`ML=0.50, none=0.30, v2=0.20`
  - 累计收益 `352.27%`
  - 年化收益 `45.40%`
  - Sharpe `1.795`
  - 最大回撤 `-20.92%`
  - 超额收益 `381.50%`
  - 超额 Sharpe `1.865`

### 结论
- `rule50` 在 `20` 股样本里更强，但在全A大样本验证里明显不如当前默认 `base`。
- 这说明上一轮 `20` 股优化结论不够稳健，不能直接迁移到执行端默认配置。
- 当前阶段更稳的选择仍然是：
  - `ML=0.70, none=0.20, v2=0.10`
- 因此执行端默认集成权重暂不修改。
- 下一步研究方向应从“继续改三路集成比例”切到：
  1. 检查全A下为何 `base` 更强
  2. 研究是否需要分市场状态使用不同的集成权重，而不是全局统一切到 `rule50`

## 2026-03-19 通达信关键信号接入：第一轮增量研究
- 目标：
  - 把通达信公式里最有价值的 6 个信号正式接入 `daily_research`
  - 检验它们是否能提升当前 advanced ML 主线的收益与超额
- 首批接入特征：
  - `kama_gap`
  - `kama_slope`
  - `long_regime_flag`
  - `mbuy_flag`
  - `zjtp_flag`
  - `hcw_flag`
- 说明：
  - 这些信号已经接入 `features.py`
  - 新增增强 profile：
    - `up_low_breakout_v3`

### 第一版直接接入结果
- 做法：
  - 连续特征和布尔信号一起进入因子库
  - 布尔信号先从 ML 输入剔除
  - 初版连续 KAMA 特征也曾进入过 ML，结果显示会拖累全A表现，因此后续做了进一步优化
- 中间结论：
  - 这类信号更适合做“规则增强层”
  - 不适合直接作为 ML 原始输入特征源去放大

### 优化后的接入方式
- 当前版本采取：
  - `kama_gap / kama_slope / *_flag`
  - 仅用于规则增强层（`up_low_breakout_v3`）
  - 不直接进入 ML 原始输入
- 原因：
  - 这些信号本质是结构事件特征
  - 稀疏度高，直接喂 ML 容易带来噪音和不稳定性

### 20 股增量 A/B
- 输出目录：
  - `daily_research/output/advanced_ml_sig3_v2_20_20260319`
  - `daily_research/output/advanced_ml_sig3_v3_20_20260319`

- `up_low_breakout_v2`
  - 累计收益 `31.06%`
  - Sharpe `0.981`
  - 超额收益 `-3.46%`
  - 超额 Sharpe `-0.102`

- `up_low_breakout_v3`
  - 累计收益 `35.36%`
  - Sharpe `1.116`
  - 超额收益 `-0.19%`
  - 超额 Sharpe `-0.005`

### 20 股结论
- 在代表性 `20` 股样本里：
  - `v3` 明显优于 `v2`
  - 说明这批信号对上涨低波增强层是有帮助的
- 但它还没有在这个样本里把超额稳定推到明显正值

### 全A验证
- 输出目录：
  - 对照基线：`daily_research/output/advanced_ml_alla_base_20220101_20260319`
  - 信号增强：`daily_research/output/advanced_ml_sig3_alla_v3_20220101_20260319`

- 对照基线 `base`
  - 累计收益 `538.81%`
  - 年化收益 `58.40%`
  - Sharpe `1.945`
  - 超额收益 `574.39%`
  - 超额 Sharpe `2.054`

- 信号增强 `up_low_breakout_v3`
  - 累计收益 `357.60%`
  - 年化收益 `45.82%`
  - Sharpe `1.490`
  - 超额收益 `390.43%`
  - 超额 Sharpe `1.640`

### 最终结论
- 这批信号不是“没用”，而是：
  - 在局部样本里能改善增强层表现
  - 但在全A主线验证里，还不足以替代当前默认主线
- 因此当前研究结论是：
  - 保留 `up_low_breakout_v3` 作为实验分支
- 暂不把它升级为 advanced ML 默认增强配置
- 下一步更合理的方向：
  1. 不再把这些信号继续硬塞进全A统一主线
  2. 考虑把它们用作：
     - `trend_up_low_vol` 状态下的候选过滤器
     - 或执行端的解释/优先级增强

## 2026-03-19 通达信关键信号接入：接入方式优化与复验
- 目标：
  - 修正第一轮接入方式中“把信号直接放进 ML”的问题
  - 重新验证这些信号在更合理接入方式下是否有增益
- 优化方式：
  - `kama_gap / kama_slope / *_flag`
  - 保留在规则增强层
  - 不再直接作为 ML 原始输入特征
- 理由：
  - 这类信号本质是结构事件
  - 稀疏度高，直接喂给 ML 容易放大噪音

### 20 股复验
- 输出目录：
  - `daily_research/output/advanced_ml_sig3_v2_20_20260319`
  - `daily_research/output/advanced_ml_sig3_v3_20_20260319`

- `up_low_breakout_v2`
  - 累计收益 `31.06%`
  - Sharpe `0.981`
  - 超额收益 `-3.46%`
  - 超额 Sharpe `-0.102`

- `up_low_breakout_v3`
  - 累计收益 `35.36%`
  - Sharpe `1.116`
  - 超额收益 `-0.19%`
  - 超额 Sharpe `-0.005`

### 20 股结论
- 优化接入方式后，`v3` 的表现明显改善。
- 这说明这批信号更适合作为增强规则层，而不是 ML 原始特征。

### 全A复验
- 输出目录：
  - `daily_research/output/advanced_ml_sig2_alla_v2_20220101_20260319`
  - `daily_research/output/advanced_ml_sig3_alla_v3_20220101_20260319`
  - 汇总表：`daily_research/output/advanced_ml_signal_compare_20260319.csv`

- `up_low_breakout_v2`
  - 累计收益 `363.72%`
  - 年化收益 `46.30%`
  - Sharpe `1.502`
  - 超额收益 `394.19%`
  - 超额 Sharpe `1.627`

- `up_low_breakout_v3`
  - 累计收益 `357.60%`
  - 年化收益 `45.82%`
  - Sharpe `1.490`
  - 超额收益 `390.43%`
  - 超额 Sharpe `1.640`

### 最终结论
- 在更合理的接入方式下，`v3` 在局部样本里优于 `v2`，说明信号本身有增益。
- 但在全A验证里，`v3` 仍未超过当前 `v2` 分支。
- 因此当前结论更新为：
  - 信号接入方向是对的
  - 接入方式必须克制
  - `up_low_breakout_v3` 保留为实验分支
  - 当前默认增强配置仍然维持 `up_low_breakout_v2`

## 2026-03-19 通达信关键信号接入：优化后最终版本
- 优化思路：
  - 第一轮结果说明，这些新信号不适合直接作为 ML 原始输入
  - 因此做了第二轮优化：
    - `kama_gap / kama_slope / *_flag`
    - 仅留在增强规则层
    - 不再直接进入 ML 特征集合
- 当前增强 profile：
  - `up_low_breakout_v3`

### 20 股优化后 A/B
- 输出目录：
  - `daily_research/output/advanced_ml_sig3_v2_20_20260319`
  - `daily_research/output/advanced_ml_sig3_v3_20_20260319`

- `up_low_breakout_v2`
  - 累计收益 `31.06%`
  - Sharpe `0.981`
  - 超额收益 `-3.46%`
  - 超额 Sharpe `-0.102`

- `up_low_breakout_v3`
  - 累计收益 `35.36%`
  - Sharpe `1.116`
  - 超额收益 `-0.19%`
  - 超额 Sharpe `-0.005`

### 20 股结论
- 优化后 `v3` 明显优于 `v2`
- 这说明这批信号在“规则增强层”里是有帮助的
- 但它们不应该直接作为 ML 原始输入特征

### 全A优化后验证
- 输出目录：
  - 对照基线：`daily_research/output/advanced_ml_alla_base_20220101_20260319`
  - 优化后增强：`daily_research/output/advanced_ml_sig3_alla_v3_20220101_20260319`

- 对照基线 `base`
  - 累计收益 `538.81%`
  - 年化收益 `58.40%`
  - Sharpe `1.945`
  - 超额收益 `574.39%`
  - 超额 Sharpe `2.054`

- 优化后增强 `up_low_breakout_v3`
  - 累计收益 `357.60%`
  - 年化收益 `45.82%`
  - Sharpe `1.490`
  - 超额收益 `390.43%`
  - 超额 Sharpe `1.640`

### 最终判断
- 这批信号在局部样本和增强层里有价值
- 但在全A统一主线验证里，仍然明显弱于当前 `base`
- 因此当前阶段：
  - 保留 `up_low_breakout_v3` 为实验分支
  - 不升级为 advanced ML 默认增强配置
- 这批信号更适合后续用于：
  1. `trend_up_low_vol` 状态下的候选过滤器
  2. 执行端建议文本中的解释增强


## 2026-03-19 更强模型对照：HistGB vs ExtraTrees（全A）
- 目的：
  - 在不改变因子、市场状态过滤、组合构建与执行链的前提下，只替换横截面模型，验证是否存在比当前 `HistGradientBoostingRegressor` 更强的可用方案。
- 说明：
  - 当前环境未安装 `LightGBM` 与 `XGBoost`，因此先使用 `sklearn` 可用的 `ExtraTreesRegressor` 作为 challenger。
- 输出目录：
  - `daily_research/output/advanced_ml_model_family_alla_20220101_20260319`
  - 汇总表：`daily_research/output/advanced_ml_model_family_alla_20220101_20260319/model_family_compare_summary.csv`

### 实验设置
- 股票池：全A，`5498` 只
- 样本区间：`2022-01-01` 至 `2026-03-19`
- 调仓频率：`5d`
- 组合目标：`holding_count=5`
- 市场状态：`trend_up_low_vol, trend_up_high_vol`
- 增强层：`up_low_breakout_v2`
- 多周期权重：`5:0.2,10:0.3,20:0.5`
- 集成权重：`ML=0.70, none=0.20, v2=0.10`

### 结果
- `histgb`
  - 累计收益 `541.64%`
  - 年化收益 `58.57%`
  - Sharpe `1.951`
  - 最大回撤 `-27.36%`
  - 超额收益 `588.47%`
  - 超额 Sharpe `2.082`

- `etr`
  - 累计收益 `238.89%`
  - 年化收益 `35.35%`
  - Sharpe `1.565`
  - 最大回撤 `-22.13%`
  - 超额收益 `263.63%`
  - 超额 Sharpe `1.645`

### 结论
- 当前全A主线下，`HistGradientBoostingRegressor` 明显优于 `ExtraTreesRegressor`。
- `ExtraTrees` 的回撤更浅，但收益、超额和风险调整后收益都明显落后。
- 因此当前 advanced ML 主线继续维持：
  - `HistGB + 现有多周期权重 + 现有集成权重`
- 这也说明：下一步如果要继续做“更强模型”研究，优先级应放在：
  1. 安装并验证 `LightGBM`
  2. 安装并验证 `XGBoost`
  3. 若仍受限于环境，再考虑更强的 `sklearn` 集成或排序目标替代方案

## 2026-03-19 更强模型对照升级：HistGB vs LightGBM（全A）
- 目的：
  - 在保持当前 advanced ML 主线结构不变的前提下，引入 `LightGBM` 做全A横截面对照，验证是否存在比 `HistGradientBoostingRegressor` 更强的可用模型。
- 输出目录：
  - `daily_research/output/advanced_ml_model_family_histgb_vs_lgbm_alla_20220101_20260319`
  - 归因目录：`daily_research/output/advanced_ml_model_family_histgb_vs_lgbm_alla_20220101_20260319/histgb_vs_lgbm`

### 实验设置
- 股票池：全A，`5498` 只
- 样本区间：`2022-01-01` 至 `2026-03-19`
- 调仓频率：`5d`
- 组合目标：`holding_count=5`
- 市场状态：`trend_up_low_vol, trend_up_high_vol`
- 增强层：`up_low_breakout_v2`
- 多周期权重：`5:0.2,10:0.3,20:0.5`
- 集成权重：`ML=0.70, none=0.20, v2=0.10`

### 结果
- `histgb`
  - 累计收益 `541.64%`
  - 年化收益 `58.57%`
  - Sharpe `1.951`
  - 最大回撤 `-27.36%`
  - 超额收益 `588.47%`
  - 超额 Sharpe `2.082`

- `lgbm`
  - 累计收益 `548.52%`
  - 年化收益 `58.99%`
  - Sharpe `1.931`
  - 最大回撤 `-22.13%`
  - 超额收益 `595.85%`
  - 超额 Sharpe `2.075`

### 归因要点
- `lgbm` 超额胜出的季度数：`7 / 17`
- 最强季度：`2025Q2`，相对 `histgb` 超额差 `+10.40%`
- 最弱季度：`2024Q4`，相对 `histgb` 超额差 `-32.78%`
- 象限对比：
  - `trend_up_low_vol`：`lgbm` 更强，超额高出约 `46.49%`
  - `trend_up_high_vol`：`histgb` 更强，超额高出约 `27.55%`
  - 下行两个象限差异很小

### 结论
- `LightGBM` 没有全方位击败 `HistGB`，但已经是当前最接近主线质量的 challenger。
- 如果优先看累计收益、超额收益和回撤，`lgbm` 有吸引力。
- 如果优先看超额 Sharpe 稳健性，当前仍然是 `histgb` 略占优。
- 因此现阶段判断：
  - 默认主线先继续保持 `histgb`
  - `lgbm` 升级为重点候选分支，下一步值得做状态内对照或双模型集成研究
## 2026-03-19 Deep Alpha 研究分支：GRU vs Transformer 烟测对照
- 目标：
  - 在新的 `daily_research/deep_alpha/` 研究分支中，不再只做静态因子加权，而是直接学习“市场状态 + 股票时序表示 + 多任务横截面排序”。
  - 第一轮先建立可运行骨架，再用两类时序编码器做最小对照：
    - `GRU`
    - `Transformer`
- 新增代码：
  - `daily_research/deep_alpha/config.py`
  - `daily_research/deep_alpha/market_state_model.py`
  - `daily_research/deep_alpha/sequence_dataset.py`
  - `daily_research/deep_alpha/models.py`
  - `daily_research/deep_alpha/trainer.py`
  - `daily_research/deep_alpha/run_deep_alpha_research.py`

### 研究设定
- 股票池：8 只代表性股票烟测
- 区间：`2022-01-01` 起
- 序列窗口：`60`
- 验证窗口：`120` 个交易日
- 目标：
  - `fwd_excess_5`
  - `fwd_excess_10`
  - `fwd_excess_20`
  - `risk_downside_20`
- 训练轮数：`2`

### 输出目录
- `daily_research/output/deep_alpha_gru_smoke_20260319`
- `daily_research/output/deep_alpha_transformer_smoke_20260319`

### GRU 结果
- 累计收益 `-4.01%`
- 年化收益 `-8.30%`
- Sharpe `-0.737`
- 超额收益 `-5.06%`
- 超额 Sharpe `-0.590`
- `fwd_excess_20 RankIC`：`0.0100`
- `risk_downside_20 RankIC`：`0.6478`

### Transformer 结果
- 累计收益 `-1.41%`
- 年化收益 `-2.96%`
- Sharpe `-0.219`
- 超额收益 `-2.49%`
- 超额 Sharpe `-0.291`
- `fwd_excess_20 RankIC`：`0.0546`
- `risk_downside_20 RankIC`：`0.5957`

### 当前结论
- `deep_alpha` 分支已经完成从数据到训练、验证、持仓打分、回测输出的完整打通。
- 在这轮烟测里，`Transformer` 明显优于 `GRU`：
  - 收益更少亏损
  - 超额更少亏损
  - `10d / 20d` 超额 RankIC 更好
- 但两者都还没有达到能挑战当前 `advanced_ml` 主线的程度。
- 这说明新分支方向是对的，但还处于“表示学习骨架刚建立、还没进入强特征/强目标/强训练范式”的阶段。

### 下一步
- 不继续做小权重扫描。
- `deep_alpha` 下一阶段优先研究：
  1. 更贴近横截面排序目标的训练损失
  2. 更强的序列特征表达
  3. 更合理的预测到持仓分数映射

## 2026-03-19 Deep Alpha 训练目标升级：按日期分组 + 排序损失
- 目标：
  - 不再只做“逐样本回归”，而是让 `deep_alpha` 更贴近真实横截面选股任务。
  - 这次升级包含两部分：
    1. 训练批次按交易日分组
    2. 在回归损失之外，引入按日期计算的成对排序损失
- 关键改动：
  - `daily_research/deep_alpha/sequence_dataset.py`
    - 新增 `DateGroupedBatchSampler`
  - `daily_research/deep_alpha/trainer.py`
    - 新增按日期的 pairwise rank loss
  - `daily_research/deep_alpha/run_deep_alpha_research.py`
    - 新增 `--ranking-loss-weight`

### 对照实验
- 输出目录：
  - `daily_research/output/deep_alpha_transformer_reg_20260319`
  - `daily_research/output/deep_alpha_transformer_rank05_20260319`

### 回归版（Transformer, ranking_loss_weight=0.0）
- 累计收益 `-6.14%`
- 超额收益 `-7.17%`
- 超额 Sharpe `-0.855`
- `fwd_excess_20 RankIC`：`0.0379`
- `fwd_excess_10 RankIC`：`-0.0893`

### 排序增强版（Transformer, ranking_loss_weight=0.5）
- 累计收益 `-1.44%`
- 超额收益 `-2.52%`
- 超额 Sharpe `-0.302`
- `fwd_excess_20 RankIC`：`0.2182`
- `fwd_excess_10 RankIC`：`0.0836`

### 结论
- 这是 `deep_alpha` 目前最有价值的一次提升。
- 排序损失没有立刻把烟测回测变成正收益，但它明显改善了核心研究指标：
  - `10d / 20d` 超额 RankIC 显著提升
  - 回测亏损明显收敛
- 这说明：
  - `deep_alpha` 当前的主要问题，已经不只是模型结构，而是目标函数和打分映射。
  - “更贴近横截面排序”这条方向是对的。

### 下一步
- 优先继续沿这条线推进，而不是回去做小参数扫描：
  1. 优化排序损失和多任务损失权重
  2. 优化预测到组合分数的映射
  3. 再做更大样本验证

## 2026-03-19 Deep Alpha 关系层第一版：行业内排名 / 行业强度 / 风格强度
- 目标：
  - 在 `deep_alpha` 中补第一版轻量关系层，不上图网络，先验证“关系信息”是否有增量。
- 接入方式：
  - 行业映射：复用 `baseline/data_provider.py` 的 `load_industry_map_from_tq`
  - 风格映射：复用 `baseline/data_provider.py` 的 `load_style_map_from_tq`
  - 新增关系特征：
    - `industry_ret_20`
    - `industry_ma_20_gap`
    - `industry_rank_ret_20`
    - `industry_rank_ma_20_gap`
    - `style_financial_strength_20`
    - `style_financial_member`
    - `style_high_dividend_strength_20`
    - `style_high_dividend_member`

### 对照实验
- 输出目录：
  - 无关系层：`daily_research/output/deep_alpha_transformer_rank05_rerun_20260319`
  - 关系层第一版：`daily_research/output/deep_alpha_transformer_rank05_rel_20260319`

### 无关系层（Transformer + rank loss）
- 特征数 `20`
- 累计收益 `1.25%`
- 超额收益 `0.14%`
- 超额 Sharpe `0.017`
- `fwd_excess_10 RankIC`：`0.1707`
- `fwd_excess_20 RankIC`：`0.1021`

### 关系层第一版
- 特征数 `28`
- 累计收益 `-0.33%`
- 超额收益 `-1.42%`
- 超额 Sharpe `-0.164`
- `fwd_excess_10 RankIC`：`0.1011`
- `fwd_excess_20 RankIC`：`0.0236`

### 当前结论
- 第一版轻量关系层已经成功接入，工程上是通的。
- 但在这轮 8 股烟测里，它明显拖累了表现。
- 这不代表“关系层方向错误”，更可能说明：
  1. 当前样本太小，关系特征噪音大于信息量
  2. 风格强度与成员特征的设计还偏粗
  3. 关系层更适合在更大横截面上验证，而不是在极小样本里判断生死

### 决策
- 关系层保留为可选开关：`--relation-layer`
- 当前 `deep_alpha` 默认研究配置仍然不启用关系层
- 下一步优先继续强化排序头，再用更大股票池重测关系层

## 2026-03-19 Deep Alpha 多任务损失权重与分数映射优化
- 目标：
  - 不再把 `5d / 10d / 20d / downside` 一视同仁。
  - 同时改进“预测值 -> 持仓分数”的映射，避免只做简单 zscore 加总。
- 工程改动：
  - `daily_research/deep_alpha/config.py`
    - 新增：
      - `target_loss_weights`
      - `score_rank_blend`
      - `score_downside_penalty`
  - `daily_research/deep_alpha/trainer.py`
    - 回归损失改为按目标加权
    - 排序损失改为按目标加权
  - `daily_research/deep_alpha/run_deep_alpha_research.py`
    - 新增参数：
      - `--task-loss-weights`
      - `--score-horizon-weights`
      - `--score-rank-blend`
      - `--score-downside-penalty`
    - 分数映射从“纯 zscore”升级为：
      - `zscore + 百分位 rank` 混合
      - 再显式扣减 downside 风险项

### 对照实验
- 目录：
  - `daily_research/output/deep_alpha_transformer_rank05_defaultscore_20260319`
  - `daily_research/output/deep_alpha_transformer_rank05_weightedscore_20260319`
  - `daily_research/output/deep_alpha_transformer_scoremap_v2_20260319`

### 基线（排序损失 + 默认损失权重 + 默认分数映射）
- 累计收益 `-1.62%`
- 超额收益 `-2.69%`
- 超额 Sharpe `-0.323`
- `fwd_excess_10 RankIC`：`0.1514`
- `fwd_excess_20 RankIC`：`0.0879`

### 方案 A（弱化 5d，强化 20d，增加 downside 惩罚）
- 参数：
  - `task-loss-weights 5:0.10,10:0.30,20:0.60,downside:0.45`
  - `score-horizon-weights 5:0.05,10:0.35,20:0.60`
  - `score-rank-blend 0.50`
  - `score-downside-penalty 0.35`
- 结果：
  - 累计收益 `-3.17%`
  - 超额收益 `-4.23%`
  - 超额 Sharpe `-0.493`

### 方案 B（更温和的 score mapping 调整）
- 参数：
  - `task-loss-weights 5:0.20,10:0.30,20:0.50,downside:0.35`
  - `score-horizon-weights 5:0.15,10:0.30,20:0.55`
  - `score-rank-blend 0.50`
  - `score-downside-penalty 0.15`
- 结果：
  - 累计收益 `-3.10%`
  - 超额收益 `-4.16%`
  - 超额 Sharpe `-0.495`

### 当前结论
- 这一步在工程上是必要的：现在 `deep_alpha` 的训练目标和打分映射终于都可配置了。
- 但就这轮 8 股烟测而言：
  - “手工指定更偏 10/20d、加强 downside 惩罚”的两版启发式权重，都没有优于当前默认配置。
- 说明当前问题不是简单把某个 horizon 权重再调大一点就能解决。
- 更可能的方向是：
  1. 用数据驱动的方法学习“预测 -> 分数”的映射
  2. 用更系统的多任务权重方案，而不是手工给定固定比例

## 2026-03-19 Deep Alpha 二层 Score Head：学习“预测 -> 持仓分数”的映射
- 目标：
  - 不再手工规定 `5d / 10d / 20d / downside` 怎么加总成最终持仓分数。
  - 改为先训练主模型，再用一个轻量二层模型学习：
    - 哪组预测组合更值得进入组合
- 新增文件：
  - `daily_research/deep_alpha/score_head.py`
- 新增能力：
  - `--score-head-method manual|ridge`
  - `--adaptive-task-weights`

### 设计
- 第一层：
  - `Transformer + 排序损失`
  - 输出：
    - `pred_fwd_excess_5`
    - `pred_fwd_excess_10`
    - `pred_fwd_excess_20`
    - `pred_risk_downside_20`
- 第二层：
  - 用训练期预测结果拟合一个 `Ridge` score head
  - 特征：
    - 每个预测值的横截面 zscore
    - 每个预测值的横截面 rank
  - 目标：
    - 由训练期 RankIC 自适应得到任务权重
    - 再构造横截面效用目标进行学习

### 对照实验
- 输出目录：
  - 手工映射：`daily_research/output/deep_alpha_scorehead_manual_20260319`
  - 学习式映射：`daily_research/output/deep_alpha_scorehead_ridge_20260319`

### 手工映射
- 累计收益 `-1.62%`
- 超额收益 `-2.69%`
- 超额 Sharpe `-0.323`
- 最大回撤 `-14.02%`

### 学习式 Score Head（Ridge + adaptive task weights）
- 累计收益 `-1.05%`
- 超额收益 `-2.13%`
- 超额 Sharpe `-0.267`
- 最大回撤 `-11.71%`

### 学到的自适应任务权重
- `fwd_excess_5`: `0.139`
- `fwd_excess_10`: `0.151`
- `fwd_excess_20`: `0.187`
- `risk_downside_20`: `0.523`

### 当前结论
- 这是 `deep_alpha` 又一个正确方向：
  - 学习式 score head 已经比手工映射更好
  - 虽然提升还不大，但方向是成立的
- 同时也暴露出一个很有价值的信息：
  - 在当前样本里，模型更依赖 `downside` 维度来形成持仓分数
  - 说明“风险维度”在当前深度分支里比我们原先设想的更重要

### 下一步
- 不再继续手工扫分数映射权重。
- 优先研究：
  1. 更强的 score head
  2. 更系统的任务权重自适应
  3. 更大样本验证

## 2026-03-19 Deep Alpha Score Head 升级：LightGBM + 窗口式自适应任务权重
- 目标：
  - 把二层 `score head` 从线性 `Ridge` 升级到更强的非线性模型。
  - 同时把 `adaptive task weights` 从“整段训练期一次性估计”推进到“最近窗口自适应”。

### 工程改动
- `daily_research/deep_alpha/score_head.py`
  - 新增 `LightGBM` score head
  - `derive_adaptive_task_weights` 支持最近窗口切片
- `daily_research/deep_alpha/run_deep_alpha_research.py`
  - 新增：
    - `--score-head-method lgbm`
    - `--adaptive-task-window-days`

### 对照实验
- 输出目录：
  - 手工映射：`daily_research/output/deep_alpha_scorehead_manual2_20260319`
  - Ridge + adaptive：`daily_research/output/deep_alpha_scorehead_ridge2_20260319`
  - LightGBM + adaptive：`daily_research/output/deep_alpha_scorehead_lgbm_20260319`

### 手工映射
- 累计收益 `-1.62%`
- 超额收益 `-2.69%`
- 超额 Sharpe `-0.323`

### Ridge + 126 日窗口式 adaptive task weights
- 累计收益 `-1.64%`
- 超额收益 `-2.72%`
- 超额 Sharpe `-0.347`

### LightGBM + 126 日窗口式 adaptive task weights
- 累计收益 `1.56%`
- 超额收益 `0.45%`
- 超额 Sharpe `0.057`

### 当前学到的窗口式 adaptive task weights
- `fwd_excess_5`: `0.271`
- `fwd_excess_10`: `0.060`
- `fwd_excess_20`: `0.101`
- `risk_downside_20`: `0.568`

### 结论
- 这一步是当前 `deep_alpha` 里最明确的一次向前推进：
  - `LightGBM score head` 已经明显优于手工映射和 `Ridge`
  - 在这轮烟测里，已经从负超额拉回到微正超额
- 同时，窗口式 adaptive task weights 也给了一个稳定信号：
  - 当前阶段 `downside` 风险维度权重最高
  - `5d` 比 `10d / 20d` 更重要
  - 这和我们之前靠直觉设定的权重并不一致

### 下一步
- 当前 `deep_alpha` 最值得继续的方向已经收敛到：
  1. 继续保留 `Transformer + rank loss`
  2. 用 `LightGBM` 做二层 score head
  3. 在更大股票池上验证这套结构

## 2026-03-19 Deep Alpha 更大股票池验证：120 只高流动性股票
- 目标：
  - 不再停留在 8 股烟测，改用更有代表性的高流动性大样本验证 `deep_alpha` 新主线。
- 股票池构建方法：
  - 从全A中取最近阶段 `20` 日平均成交额最高的 `120` 只股票
  - 文件：
    - `daily_research/output/deep_alpha_liquid120_20260319.txt`

### 实验结构
- 主模型：
  - `Transformer + rank loss`
- 对照：
  1. 手工映射：`manual`
  2. 二层头：`LightGBM score head + 126日 adaptive task weights`

### 输出目录
- 手工映射：
  - `daily_research/output/deep_alpha_liquid120_manual_20260319`
- LightGBM score head：
  - `daily_research/output/deep_alpha_liquid120_lgbm_20260319`

### 手工映射结果
- 训练样本 `56225`
- 验证样本 `10469`
- 累计收益 `50.30%`
- 年化收益 `136.99%`
- 超额收益 `48.66%`
- 超额 Sharpe `4.011`
- 最大回撤 `-18.84%`

### LightGBM score head 结果
- 训练样本 `56225`
- 验证样本 `10469`
- 累计收益 `8.68%`
- 年化收益 `19.26%`
- 超额收益 `7.49%`
- 超额 Sharpe `0.813`
- 最大回撤 `-12.59%`

### 当前结论
- 这是一个很重要的修正：
  - `LightGBM score head` 在 8 股烟测里优于手工映射
  - 但在 120 只高流动性股票样本里，明显弱于手工映射
- 这说明：
  1. 8 股烟测里的 `LightGBM` 改善具有明显样本局部性
  2. 当前二层头还不够稳，不适合立刻升成默认主线
  3. 更大样本验证是必要的，而且已经帮我们避免了错误升级

### 补充观察
- 这轮大样本里，第一层多任务预测的 return 维度 RankIC 本身是负的，只有 `risk_downside_20` 很强。
- 手工映射之所以能跑得更好，更可能是：
  - 当前 `deep_alpha` 的有效信息主要还集中在“风险规避”
  - 手工映射保留了更直接的风险主导逻辑
  - `LightGBM score head` 目前还没学出稳定的横截面选股增强

### 决策
- 当前阶段：
  - `deep_alpha` 保留 `Transformer + rank loss`
  - 二层 `score head` 继续作为实验层
  - 默认不切换到 `LightGBM score head`
- 下一步优先不再盲目换头，而是回到更本质的问题：
  1. 第一层 return 预测为什么在更大样本里变弱
  2. 如何让第一层学到更稳定的 return 排序信息


## 2026-03-19 Deep Alpha 第一层多任务目标失效结构分析（120 高流动性股票池）
- 目标：
  - 明确第一层多任务目标在更大样本里到底是哪里失效：
    1. 哪个 horizon 在拖后腿
    2. 哪些市场状态里 return 预测变差
    3. 是否已经变成“只会避险，不会进攻”
- 新增分析脚本：
  - `daily_research/deep_alpha/analyze_target_failure.py`
- 分析对象：
  - `daily_research/output/deep_alpha_liquid120_manual_20260319`
- 输出目录：
  - `daily_research/output/deep_alpha_liquid120_manual_20260319/target_failure_analysis`

### 关键结果
#### 1. 拖后腿的 horizon
- 第一层 return 目标在大样本里整体都偏弱或为负：
  - `fwd_excess_5 RankIC = -0.0316`
  - `fwd_excess_10 RankIC = -0.0292`
  - `fwd_excess_20 RankIC = -0.0618`
- 明确最差的是：
  - `fwd_excess_20`

#### 2. 哪些市场状态里 return 预测变差
- `trend_up_low_vol`
  - `fwd_excess_20 RankIC = -0.0821`
  - `fwd_excess_5 RankIC = -0.0391`
  - `fwd_excess_10 RankIC = -0.0031`
- `trend_down_low_vol`
  - `fwd_excess_20 RankIC = -0.0451`
  - `fwd_excess_10 RankIC = -0.0505`
  - `fwd_excess_5 RankIC = -0.0254`

结论：
- 当前验证窗口里，第一层 return 学习在低波状态下整体失效；
- 其中最明显的失效区是 `trend_up_low_vol`，也就是最该学会“进攻”的地方。

#### 3. 是否变成“只会避险，不会进攻”
- 是，而且证据很强。
- 最终手工映射分数与未来 20 日超额收益的日度平均秩相关：
  - `-0.0581`
- 最终手工映射分数与未来 20 日 downside 风险的日度平均秩相关：
  - `0.4548`
- 最高分组 vs 最低分组：
  - `20d` 真实超额收益差：`-0.0476`
  - downside 风险差：`+0.1536`

这说明：
- 当前分数确实更擅长挑出“更安全的股票”
- 但没有把“未来 20 日更强的超额收益”排到前面
- 本质上已经更像风险模型，而不是攻击型 alpha 模型

### 进一步理解
- 当前 120 股票池里，手工映射仍然能跑出不错的组合结果，不是因为第一层 return 学得好了，
  更可能是因为：
  1. `risk_downside_20` 非常强
  2. 手工映射保留了更直接的风险规避逻辑
  3. 组合收益更多来自“少踩坑”，不是“更精准地抓住超额赢家”

### 决策
- 现在不优先继续微调二层 `score head`
- 研究重点应转向：
  1. **强化第一层 return 学习**
  2. 必要时再**改变目标定义**

当前优先判断：
- 第一优先级：先强化第一层 return 学习
- 因为现阶段最核心的问题，不是“怎么把几个预测值加起来”，而是“第一层对 return 的学习本身就不够对题”

### 下一步方向
建议下一步聚焦以下两条：
1. 重新设计第一层 return 目标
   - 从直接回归未来超额收益，转向更贴近横截面排序的目标
   - 例如分层标签、top-vs-rest 分类、排序型 target
2. 把 risk head 与 return head 的职责进一步拆开
   - risk 负责过滤和惩罚
   - return 负责决定谁值得进攻
   - 避免当前这种“risk 太强，把进攻信号全部盖住”的结构

## 2026-03-19 Deep Alpha 风险门控升级：固定 gate vs 状态自适应 gate（300 高流动性子样本）
- 目标：
  - 把 `risk gate` 从固定阈值升级成可学习/按状态自适应；
  - 并在更大股票池上验证它是否真的值得进入 `deep_alpha` 默认主线。

### 代码改动
- `daily_research/deep_alpha/risk_gate.py`
- `daily_research/deep_alpha/run_deep_alpha_research.py`

新增能力：
- `--score-risk-mode state_gate`
- `--score-risk-state-thresholds`
- 训练期自动学习各市场状态下的最优 risk gate 阈值
- 运行后会额外输出：
  - `risk_gate_objective_rows.csv`

### 验证样本
- 股票池：
  - `daily_research/output/deep_alpha_liquid300_20260319.txt`
- 规模：
  - `300` 只高流动性 A 股
- 区间起点：
  - `2024-01-01`
- 验证窗口：
  - `126` 个交易日
- 主模型：
  - `Transformer + rank loss`
- 第一层 return 目标：
  - `regression`

### 对比结果
#### 1. 固定 subtract（当前旧式风险惩罚）
- 输出：
  - `daily_research/output/deep_alpha_liquid300_regression_subtract_20260319`
- 结果：
  - 累计收益 `90.74%`
  - 超额收益 `85.63%`
  - 超额 Sharpe `7.321`

#### 2. 固定 gate 0.35
- 输出：
  - `daily_research/output/deep_alpha_liquid300_regression_gate35_20260319`
- 结果：
  - 累计收益 `38.05%`
  - 超额收益 `34.35%`
  - 超额 Sharpe `2.426`

#### 3. 状态自适应 gate
- 输出：
  - `daily_research/output/deep_alpha_liquid300_regression_stategate_20260319`
- 结果：
  - 累计收益 `97.52%`
  - 超额收益 `92.23%`
  - 超额 Sharpe `8.012`

### 学到的 gate 阈值
- 全局最优阈值：
  - `0.0`
- 状态阈值：
  - `trend_down_low_vol -> 0.0`
  - `trend_up_high_vol -> 0.0`

### 结论
- 这轮结果说明两件事：
  1. 固定 `gate=0.35` 过强，明显伤害收益
  2. 状态自适应 gate 比固定 subtract 和固定 gate 都更好

但更关键的是：
- 自适应 gate 最终学到的不是“更强过滤”，而是“当前这组训练样本里不该强行过滤”
- 也就是说，这套机制的价值在于：
  - **让 risk 真的变成一个可学习的约束层**
  - 而不是硬编码一个固定阈值去压制 return

### 决策
- `deep_alpha` 当前最优研究配置更新为：
  - 第一层：`regression + rank loss`
  - 分数层：`state_gate`
- 但这还不足以直接升级成整个项目的默认执行主线，因为：
  - 目前验证的是 `300` 只高流动性子样本
  - 还不是全A更大范围验证

### 当前判断
- 可以把 `state_gate` 升级为 **deep_alpha 研究默认配置**
- 但还不能直接升为 **daily execution 默认主线**
- 下一步应继续：
  1. 用更大股票池 / 更广泛全A子样本复验
2. 如果结果继续稳定，再讨论是否进入执行端主线

## 2026-03-19 Deep Alpha 扩展验证：500 高流动性股票池上的 state_gate 稳健性
- 目标：
  - 不再停留在 `300` 只股票子样本；
  - 继续放大到 `500` 只高流动性股票池，验证 `state_gate` 是否还能保持优势。

### 验证样本
- 股票池文件：
  - `daily_research/output/deep_alpha_liquid500_20260319.txt`
- 样本规模：
  - `500` 只高流动性 A 股
- 区间起点：
  - `2024-01-01`
- 验证窗口：
  - `126` 个交易日
- 主模型：
  - `Transformer + rank loss`
- 第一层 return：
  - `regression`

### 对照实验
#### 1. 固定 subtract
- 输出：
  - `daily_research/output/deep_alpha_liquid500_regression_subtract_20260319`
- 结果：
  - 累计收益 `39.47%`
  - 超额收益 `35.73%`
  - 超额 Sharpe `2.503`

#### 2. 状态自适应 gate
- 输出：
  - `daily_research/output/deep_alpha_liquid500_regression_stategate_20260319`
- 结果：
  - 累计收益 `61.46%`
  - 超额收益 `57.13%`
  - 超额 Sharpe `4.058`

### 学到的 gate 阈值
- 全局阈值：
  - `0.0`
- 状态阈值：
  - `trend_down_low_vol -> 0.0`
  - `trend_up_high_vol -> 0.5`

### 结论
- 这轮 500 股票池验证说明：
  - `state_gate` 不是只在 300 股票池里偶然有效
  - 在更大的高流动性子样本里，它仍然明显优于固定 subtract
- 更重要的是，这次它不再退化成“全局都不 gate”：
  - 在 `trend_up_high_vol` 里，模型主动学到了 `0.5` 的更强风控阈值
  - 在 `trend_down_low_vol` 里，仍保持 `0.0`

这说明：
- `state_gate` 已经开始表现出真正的“按状态自适应”
- 而不是简单地退化成无过滤

### 当前判断
- 可以把：
  - `regression + rank loss + state_gate`
 视为当前 `deep_alpha` 更稳定、更可信的研究主线
- 但仍然不建议直接升为执行端默认，因为：
  - 现在的证据仍集中在高流动性全A子样本
  - 还缺一次更广的全A覆盖验证

### 下一步
优先顺序建议：
1. 用更广的全A子样本继续复验 `state_gate`
2. 如果仍保持优势，再考虑把它提升为 `deep_alpha` 默认主线
3. 最后再讨论它是否接近执行端

## 2026-03-19 Deep Alpha 第一层目标重做：return 分类化 + risk/return 解耦
- 目标：
  - 不再只沿用“直接回归未来超额收益”的第一层目标；
  - 尝试把 return 目标改成更贴近横截面排序的形式；
  - 同时把 risk 从“主导分数”改成更清晰的过滤/约束角色。

### 代码改动
- `daily_research/deep_alpha/config.py`
- `daily_research/deep_alpha/trainer.py`
- `daily_research/deep_alpha/run_deep_alpha_research.py`

新增能力：
- `--return-loss-mode`
  - `regression`
  - `top_rest_bce`
  - `top_bottom_bce`
- `--return-top-frac`
- `--return-bottom-frac`
- `--score-risk-mode`
  - `subtract`
  - `gate`
- `--score-risk-gate-threshold`

### 验证样本
- 股票池：
  - `daily_research/output/deep_alpha_liquid120_20260319.txt`
- 区间起点：
  - `2024-01-01`
- 验证窗口：
  - `126` 个交易日
- 主模型：
  - `Transformer + rank loss`
- 二层：
  - `manual score head`

### A/B 结果
#### 1. 第一层直接回归 return（baseline）
- 输出：
  - `daily_research/output/deep_alpha_liquid120_regression_retarget_20260319`
- 结果：
  - 累计收益 `85.47%`
  - 超额收益 `80.49%`
  - 超额 Sharpe `6.515`
  - `fwd_excess_5 RankIC = 0.0610`
  - `fwd_excess_10 RankIC = 0.0711`
  - `fwd_excess_20 RankIC = 0.0555`

#### 2. 第一层改成 top-bottom 分类
- 输出：
  - `daily_research/output/deep_alpha_liquid120_topbottom_retarget_20260319`
- 结果：
  - 累计收益 `28.40%`
  - 超额收益 `24.95%`
  - 超额 Sharpe `2.751`
  - `fwd_excess_5 RankIC = -0.0176`
  - `fwd_excess_10 RankIC = -0.0044`
  - `fwd_excess_20 RankIC = 0.0000`

#### 3. 第一层改成 top-vs-rest 分类
- 输出：
  - `daily_research/output/deep_alpha_liquid120_toprest_retarget_20260319`
- 结果：
  - 累计收益 `72.93%`
  - 超额收益 `68.29%`
  - 超额 Sharpe `5.891`
  - `fwd_excess_5 RankIC = 0.0251`
  - `fwd_excess_10 RankIC = 0.0090`
  - `fwd_excess_20 RankIC = -0.0065`

#### 4. 保留回归型 return，但把 risk 从“减分”改成“过滤”
- 输出：
  - `daily_research/output/deep_alpha_liquid120_regression_gate35_20260319`
- 设定：
  - `return_loss_mode = regression`
  - `score_risk_mode = gate`
  - `score_risk_gate_threshold = 0.35`
- 结果：
  - 累计收益 `76.51%`
  - 超额收益 `71.77%`
  - 超额 Sharpe `7.195`
  - 超额最大回撤 `-5.99%`

### 结论
- 这轮结果非常明确：
  1. 在当前 120 股票池验证窗口里，**直接把第一层 return 目标改成分类，并没有打赢回归**
  2. `top_bottom_bce` 明显最差，说明它丢掉了太多中间样本信息
  3. `top_rest_bce` 比 `top_bottom_bce` 好，但仍然弱于回归
  4. **真正有效的是 risk / return 职责拆开**

更具体地说：
- 当前这版 `deep_alpha` 第一层最好的 return 学习方式，仍然是：
  - `regression + rank loss`
- 但分数层不该继续让 risk 与 return 混成一团：
  - `risk` 更适合做过滤器
  - `return` 更适合决定谁值得进入组合

### 决策
- 当前不把第一层默认切到分类目标
- 当前最值得保留的新结论是：
  - `return_loss_mode = regression`
  - `score_risk_mode = gate`
- 下一步研究重点应转向：
  1. 在保留 `regression + rank loss` 的前提下，继续增强第一层 return 学习
  2. 把 risk gating 做成更稳的可学习或可验证机制

## 2026-03-19 deep_alpha 提速改造：缓存 + 单次构建语料 + 排序损失采样

### 背景
- `deep_alpha` 在更大股票池验证时，主要耗时已经不只是训练本身，而是：
  1. 每次都重新从 TQ 拉取原始日线
  2. `train_ds / valid_ds` 分别双重构建样本
  3. `pairwise rank loss` 对同日样本做全量成对比较

### 本次改动
- 新增缓存模块：
  - `daily_research/deep_alpha/cache_utils.py`
- `run_deep_alpha_research.py` 现在支持：
  - 原始 TQ 数据缓存
  - 特征/目标缓存
  - `--no-cache`
  - `--refresh-cache`
  - `--num-workers`
  - `--pin-memory`
  - `--use-amp`
  - `--max-rank-pairs-per-group`
- `sequence_dataset.py` 改成：
  - 先一次性构建 `StockSequenceCorpus`
  - 再切分 `train / valid` 视图
  - 避免原先双重构建数据集
- `trainer.py` 改成：
  - 支持 `AMP` 混合精度
  - 推理阶段也支持 `AMP`
  - 排序损失支持 `max_rank_pairs_per_group` 采样，避免全量 `O(n^2)` pair 计算

### 小规模验证
- 运行：
  - `daily_research/output/deep_alpha_speed_smoke_small3_20260319`
- 条件：
  - `5` 只股票
  - `Transformer`
  - `epochs=1`
  - `num_workers=2`
  - `AMP on`
- 结果：
  - 流程已完整跑通
  - 原始数据缓存命中正常
  - 特征缓存命中正常
  - 单次构建语料 + 视图切分正常

### 当前判断
- 这轮改动属于高性价比的工程提速，不改变策略逻辑。
- 目前 `deep_alpha` 研究端已经具备三层提速能力：
  1. 原始数据缓存
  2. 特征/目标缓存
  3. 排序损失采样 + GPU 混合精度
- 下一步如果还要继续提速，优先级应是：
  1. 样本语料缓存
  2. 更高效的 pair 采样策略
  3. 分阶段增量更新而不是全量重算

## 2026-03-19 Deep Alpha vs Advanced ML：500 高流动性股票池同池同区间正面对照

### 目标
- 用同一股票池、同一区间，直接回答：
  - `deep_alpha` 目前只是“研究上有希望”
  - 还是已经开始“接近替代执行主线”

### 对照设置
- 股票池：
  - `daily_research/output/deep_alpha_liquid500_20260319.txt`
- 区间起点：
  - `2024-01-01`
- 对照模型：
  - `advanced_ml`
    - `HistGB`
    - 多周期 `5/10/20 = 0.2/0.3/0.5`
    - 集成权重 `ml=0.70, none=0.20, v2=0.10`
  - `deep_alpha`
    - `Transformer + rank loss + regression + state_gate`
- 为公平起见，最终对照窗口统一到：
  - `2025-09-05` 至 `2026-03-19`
  - 即 `deep_alpha` 当前 500 股票验证的 holdout 窗口

### 输出
- `advanced_ml`：
  - `daily_research/output/advanced_ml_liquid500_compare_20260319`
- `deep_alpha`：
  - `daily_research/output/deep_alpha_liquid500_regression_stategate_20260319`
- 对照汇总：
  - `daily_research/output/deepalpha_vs_advancedml_liquid500_20260319.csv`

### 对照结果（统一 holdout 口径）
#### 1. Advanced ML
- 累计收益：`66.67%`
- 超额收益：`58.75%`
- 超额 Sharpe：`4.338`
- 超额最大回撤：`-10.28%`
- 平均持仓数：`3.89`
- 平均换手：`1.119`
- 胜率：`45.24%`

#### 2. Deep Alpha state_gate
- 累计收益：`61.46%`
- 超额收益：`57.13%`
- 超额 Sharpe：`4.058`
- 超额最大回撤：`-16.73%`
- 平均持仓数：`4.18`
- 平均换手：`0.853`
- 胜率：`47.62%`

### 结论
- 这轮最重要的结论是：
  - `deep_alpha` 已经不再是“明显落后”的实验分支
  - 但在当前这轮 500 股票、同池同区间的正面对照下，仍未打赢当前 `advanced_ml` 执行主线
- 更细一点看：
  - `deep_alpha` 的优势：
    - 换手更低
    - 胜率略高
  - `advanced_ml` 的优势：
    - 收益更高
    - 超额 Sharpe 更高
    - 回撤更浅

### 当前决策
- **暂不把 `deep_alpha` 接入执行端主线**
- **执行端继续保持 `advanced_ml` 为默认主线**
- `deep_alpha` 当前定位更新为：
  - 已经进入“接近执行主线”的候选阶段
  - 但还需要继续做更本质的结构学习，而不是贸然接管执行

### 下一步研究方向
1. 如果继续推进 `deep_alpha`
   - 优先做更强的第一层 `return` 学习
   - 而不是继续折腾 profile 或小参数
2. 如果要继续接近执行端
   - 先扩大到更广的全A子样本，重复这类同池同区间对照
3. 在此之前
   - `advanced_ml` 继续作为稳定执行主线

## 2026-03-19 Deep Alpha 原始 return 学习升级：更贴近 winner-picking 排序目标

### 目标
- 之前的 `deep_alpha` 更偏向“风险约束 + 回归拟合”，第一层 `return` 学习还不够直接服务于挑选下一阶段更强股票。
- 这一步的目标是让 `return` 学习更贴近横截面 winner-picking，而不是继续把主信号交给后处理阶段补救。

### 本次改动
- `daily_research/deep_alpha/models.py`
  - 进一步拆分 `return head` 与 `risk head`
- `daily_research/deep_alpha/sequence_dataset.py`
  - 调整标签与样本组织方式，让 `return` 目标更直接服务排序
- `daily_research/deep_alpha/trainer.py`
  - 新增 `listwise rank loss`
- `daily_research/deep_alpha/run_deep_alpha_research.py`
  - 新增参数：
    - `--return-target-transform`
    - `--listwise-loss-weight`
    - `--listwise-temperature`

### 结果
#### 300 高流动性股票池
输出：
- `daily_research/output/deep_alpha_liquid300_regression_stategate_20260319`
- `daily_research/output/deep_alpha_liquid300_raw_listwise25_e4_20260319`
- `daily_research/output/deep_alpha_liquid300_csrank_listwise25_e4_20260319`

结果：
- 原始 `state_gate`
  - 超额收益：`92.23%`
  - 超额 Sharpe：`8.01`
- `raw + listwise`
  - 超额收益：`134.52%`
  - 超额 Sharpe：`10.10`
- `cs_rank + listwise`
  - 超额收益：`73.96%`
  - 超额 Sharpe：`6.27`

#### 500 高流动性股票池
输出：
- `daily_research/output/deep_alpha_liquid500_regression_stategate_20260319`
- `daily_research/output/deep_alpha_liquid500_raw_listwise25_e4_20260319`

结果：
- 原始 `state_gate`
  - 超额收益：`57.13%`
  - 超额 Sharpe：`4.06`
- `raw + listwise`
  - 超额收益：`193.79%`
  - 超额 Sharpe：`17.20`

### 结论
- 当前证据很清楚：要让第一层 `return target` 更直接服务 `rank learning`。
- 目前最优方向是：
  - 使用 `raw return target`
  - 叠加 `pairwise + listwise` 排序损失
  - 再结合 `state_gate`
- 因此 `deep_alpha` 的第一层正式研究主线更新为：
  - `raw return target`
  - `pairwise rank loss`
  - `listwise rank loss`
  - `state_gate`

### 汇总输出
- `daily_research/output/deep_alpha_return_learning_compare_20260319.csv`
- `daily_research/output/deep_alpha_return_learning_vs_advancedml_20260319.csv`

## 2026-03-20 Deep Alpha 严格复验：500 高流动性股票池 + 更长 252 日 holdout

### 目的
- 不再只看短窗口结果，而是用更长 holdout 检查 `deep_alpha` 新主线是否仍然成立。
- 在同一股票池、同一时间窗口下，和当前执行主线 `advanced_ml` 正面对照。

### 设置
- 股票池：`daily_research/output/deep_alpha_liquid500_20260319.txt`
- 研究起点：`2024-01-01`
- 对照窗口：`2025-03-06` 到 `2026-03-19`
- `deep_alpha` 配置：
  - `Transformer`
  - `regression`
  - `pairwise rank loss = 0.5`
  - `listwise rank loss = 0.25`
  - `state_gate`

### 输出
- `deep_alpha`
  - `daily_research/output/deep_alpha_liquid500_raw_listwise25_e4_v252_20260320`
- `advanced_ml` 同窗口切片
  - `daily_research/output/advanced_ml_liquid500_compare_20260319/metrics_holdout_20250306_20260319.json`
- 汇总
  - `daily_research/output/deep_alpha_strict_revalidation_20260320.csv`

### 结果
#### Advanced ML
- 超额收益：`54.80%`
- 超额 Sharpe：`1.643`
- 超额最大回撤：`-16.76%`

#### Deep Alpha 新主线（252 日 holdout）
- 超额收益：`83.09%`
- 超额 Sharpe：`2.746`
- 超额最大回撤：`-14.44%`

### 结论
- 这次已经不是短窗口偶然。`deep_alpha` 新主线在更长 holdout 下仍然赢过了 `advanced_ml`。
- 更关键的是，它同时做到了：
  - 更高的超额收益
  - 更高的风险调整后收益
  - 更浅的超额回撤

### 决策
- `deep_alpha` 已经进入“接近执行主线”的候选阶段。
- 但还不直接接入执行端，下一步先做更广高流动性子样本复验。

## 2026-03-20 Deep Alpha 更广子样本复验：800 高流动性股票池同池同区间对照

### 目的
- 在比 `liquid500` 更广的高流动性子样本上，继续检验 `deep_alpha` 新主线是否成立。
- 若仍保持优势，再决定是否进入执行前最后对照阶段。

### 这次补充改动
- `daily_research/baseline/run_advanced_daily_research.py`
  - 新增 `--stocks-file`，方便 `advanced_ml` 复用同一股票池做严格对照。

### 设置
- 股票池生成方式：从全A按最近 `ADV20` 选取前 `800` 只高流动性股票
- 股票池文件：
  - `daily_research/output/deep_alpha_liquid800_20260320.txt`
- `deep_alpha` 配置保持不变：
  - `raw return target`
  - `pairwise rank loss = 0.5`
  - `listwise rank loss = 0.25`
  - `state_gate`
- `advanced_ml` 使用同一股票池、同一研究起点、同一 holdout 区间

### 输出
- `deep_alpha`
  - `daily_research/output/deep_alpha_liquid800_raw_listwise25_e4_v252_20260320`
- `advanced_ml`
  - `daily_research/output/advanced_ml_liquid800_compare_20260320`
  - `daily_research/output/advanced_ml_liquid800_compare_20260320/metrics_holdout_20250307_20260319.json`
- 执行前稳定性对照汇总
  - `daily_research/output/deep_alpha_execution_readiness_compare_20260320.csv`

### 结果
#### Deep Alpha
- 超额收益：`149.60%`
- 超额 Sharpe：`3.508`
- 超额最大回撤：`-28.55%`
- 平均换手：`0.927`
- 胜率：`51.59%`

#### Advanced ML
- 超额收益：`88.39%`
- 超额 Sharpe：`1.669`
- 超额最大回撤：`-21.41%`
- 平均换手：`1.143`
- 胜率：`41.27%`

### 执行前稳定性观察
- `deep_alpha` 在 `liquid500` 和 `liquid800` 两个更大高流动性样本上都保持了超额优势。
- `deep_alpha` 的共同特点是：
  - 超额收益更高
  - 超额 Sharpe 更高
  - 胜率更高
  - 在 `liquid800` 上换手还更低
- 需要继续关注的一点：
  - `liquid800` 上 `deep_alpha` 的超额回撤更深，说明它已经开始更强地参与进攻排序，后面仍需继续监控回撤稳定性。

### 当前决策
- `advanced_ml` 仍然保留为当前执行端默认主线。
- `deep_alpha` 已经不再只是研究备选，而是进入“执行前最后验证阶段”的候选主线。
- 下一步研究继续围绕：
  1. 更广高流动性或分层全A子样本复验
  2. 执行前最后对照：稳定性、换手、回撤一致性
  3. 若优势继续保持，再讨论接近执行端
## 2026-03-20 Deep Alpha 分层全A子样本复验：确认是否只在高流动性样本里强
### 目的
- 不再只看高流动性股票池，转而用“分层全A子样本”验证 `deep_alpha`。
- 这一步要回答的不是收益能不能继续做高，而是：
  - `deep_alpha` 的优势是否只集中在高流动性样本里
  - 它离真正可泛化的执行主线还有多远

### 样本构造
- 先从全A里按最近 `ADV20` 做 5 档流动性分层。
- 每一档等量抽取 `160` 只股票，总共 `800` 只。
- 股票池文件：
  - `daily_research/output/deep_alpha_stratified_alla800_20260320.txt`
- 分层摘要：
  - `daily_research/output/deep_alpha_stratified_alla800_20260320_summary.csv`

### 对照设置
- `deep_alpha`
  - `raw return target`
  - `pairwise rank loss = 0.5`
  - `listwise rank loss = 0.25`
  - `state_gate`
- `advanced_ml`
  - 当前执行端默认主线
- 时间窗口统一为长 holdout：
  - `2025-03-07` 到 `2026-03-20`

### 输出
- `deep_alpha`
  - `daily_research/output/deep_alpha_stratified_alla800_raw_listwise25_e4_v252_20260320`
- `advanced_ml`
  - `daily_research/output/advanced_ml_stratified_alla800_compare_20260320`
  - `daily_research/output/advanced_ml_stratified_alla800_compare_20260320/metrics_holdout_20250307_20260320.json`
- 汇总
  - `daily_research/output/deep_alpha_stratified_alla800_compare_20260320.csv`

### 结果
#### Deep Alpha
- 超额收益：`0.22%`
- 超额 Sharpe：`0.007`
- 超额最大回撤：`-28.84%`
- 平均换手：`0.641`
- 胜率：`44.84%`

#### Advanced ML
- 超额收益：`85.24%`
- 超额 Sharpe：`1.138`
- 超额最大回撤：`-16.15%`
- 平均换手：`1.087`
- 胜率：`39.68%`

### 结论
- 这次结果非常关键：`deep_alpha` 在分层全A子样本上没有保持住优势。
- 它在高流动性子样本上赢过 `advanced_ml`，但一旦把中低流动性层也系统性纳入，表现几乎被抹平。
- 这说明当前 `deep_alpha` 的有效性高度依赖流动性环境，暂时还不能视为“全A可泛化主线”。

### 阶段判断
- 现在不应推动 `deep_alpha` 接近执行端。
- 当前最稳的结论是：
  - `advanced_ml` 继续保留为执行端默认主线
  - `deep_alpha` 回到“研究主线”定位
- 下一步研究应转向：
  1. 明确建模“流动性条件”本身
  2. 研究 `deep_alpha` 为何只在高流动性样本里有效
  3. 考虑把流动性分层作为显式状态或关系特征接入

## 2026-03-20 流动性条件显式研究：deep_alpha 在分层全A子样本上的复验

### 这次做了什么
- 把流动性条件提升为 `deep_alpha` 的显式研究对象。
- 新增了两类能力：
  - `liquidity_layer`：把流动性分层状态和流动性相关关系特征接进序列特征。
  - `state_liquidity_gate`：让 risk gate 按“市场状态 + 流动性桶”联合学习阈值。
- 在分层全A `800` 股票池上做了同口径验证，避免只看高流动性股票池的结果。

### 关键输出
- 结果汇总：`daily_research/output/deep_alpha_liquidity_research_compare_20260320.csv`
- 流动性桶诊断：`daily_research/output/deep_alpha_stratified_alla800_liquidity_bucket_diagnostics_20260320.csv`
- 运行结果：
  - `daily_research/output/deep_alpha_stratified_alla800_raw_listwise25_e4_v252_20260320`
  - `daily_research/output/deep_alpha_stratified_alla800_liqgateonly_20260320`
  - `daily_research/output/deep_alpha_stratified_alla800_liqfeatureonly_20260320`
  - `daily_research/output/deep_alpha_stratified_alla800_liquidityaware_20260320`
  - `daily_research/output/advanced_ml_stratified_alla800_compare_20260320`

### 对照结果
- `deep_alpha baseline state_gate`
  - 超额收益：`0.22%`
  - 超额 Sharpe：`0.007`
- `deep_alpha + state_liquidity_gate`
  - 超额收益：`0.22%`
  - 超额 Sharpe：`0.007`
- `deep_alpha + liquidity_layer`
  - 超额收益：`-21.40%`
  - 超额 Sharpe：`-0.641`
- `deep_alpha + liquidity_layer + state_liquidity_gate`
  - 超额收益：`-21.40%`
  - 超额 Sharpe：`-0.641`
- `advanced_ml`（同 holdout）
  - 超额收益：`85.24%`
  - 超额 Sharpe：`1.138`

### 结论
- `state_liquidity_gate` 单独启用时没有带来改进，学习出的分层阈值基本退化为 `0.0`，说明当前 gate 还没有学到有用的流动性过滤规则。
- `liquidity_layer` 直接作为一层额外特征接进模型后，表现显著变差，说明“把流动性原始分层特征直接堆到特征栈里”不是当前最有效的接法。
- 这次结果进一步确认：`deep_alpha` 的有效性仍然明显依赖高流动性环境，暂时还不是可泛化到更广分层全A样本的主线。
- 当前执行端默认主线继续保持 `advanced_ml`，`deep_alpha` 继续作为研究主线推进。

### 新的研究判断
- 流动性条件必须继续作为显式研究对象，但下一步不应再沿着“直接加原始流动性特征 + 直接做 state_liquidity_gate”这条线硬推。
- 更有前景的方向是：
  1. 先做更细的流动性失效诊断，确认哪些流动性层、哪些状态组合在拖后腿。
  2. 把流动性作为条件变量，而不是简单拼进主特征栈。
  3. 优先研究“按流动性条件切换 return 学习”，而不是先继续强化 risk gate。

## 2026-03-20 流动性失效细诊断 + 条件化 return 学习第一版

### 这次做了什么
- 给 `deep_alpha` 增加了按流动性桶切换的 `return head`：
  - `--return-head-mode liquidity_switch`
- 新增流动性失效诊断脚本：
  - `daily_research/deep_alpha/analyze_liquidity_failure.py`
- 在分层全A `800` 子样本上，对比了：
  - `shared_state_gate`
  - `liquidity_switch_state_gate`
  - `advanced_ml`

### 关键输出
- 对照汇总：`daily_research/output/deep_alpha_liquidity_condition_compare_20260320.csv`
- 基线 run：
  - `daily_research/output/deep_alpha_stratified_alla800_shared_liqdiag_20260320`
- 条件化 return head run：
  - `daily_research/output/deep_alpha_stratified_alla800_liqswitch_20260320`
- 诊断输出：
  - `daily_research/output/deep_alpha_stratified_alla800_shared_liqdiag_20260320/liquidity_failure_analysis`
  - `daily_research/output/deep_alpha_stratified_alla800_liqswitch_20260320/liquidity_failure_analysis`

### 结果
#### Shared return head
- 超额收益：`0.22%`
- 超额 Sharpe：`0.007`
- 在流动性桶上的 `fwd_excess_20 RankIC`
  - `bucket 3`：`-0.019`
  - `bucket 4`：`0.094`

#### Liquidity-switch return head
- 超额收益：`-26.58%`
- 超额 Sharpe：`-0.628`
- 在流动性桶上的 `fwd_excess_20 RankIC`
  - `bucket 3`：`-0.149`
  - `bucket 4`：`-0.041`

#### Advanced ML（同 holdout）
- 超额收益：`85.24%`
- 超额 Sharpe：`1.138`

### 结论
- 更细的流动性诊断说明：即使在已经做过 `min_adv20` 过滤的分层全A子样本里，`deep_alpha` 的第一层 return 学习也主要只在最高流动性桶里有效。
- 把流动性当条件变量这个方向是对的，但当前这版“按每个流动性桶切换独立 return head”过于粗暴，明显伤害了泛化。
- 所以这一步的价值不是把 `liquidity_switch` 升成新主线，而是进一步确认：
  - 流动性问题的核心在第一层 return 学习
  - 下一步应优先研究“更平滑、更约束的流动性条件化 return 学习”，而不是直接把桶切得很硬

### 下一步收敛
- 暂不采用 `liquidity_switch` 作为默认研究配置。
- 更值得做的是：
  1. 先把流动性条件压缩成更少的 regime，例如“最高流动性 vs 其余”。
  2. 让流动性只影响 return loss / sample weighting，而不是直接切独立 head。
  3. 继续保持 `advanced_ml` 为执行端默认主线，`deep_alpha` 聚焦研究。

## 2026-03-20 执行与研究假设改造：盘后出策略，次日开盘执行

### 背景
- 之前执行端更偏“同日尾盘执行”，而缓存又按日期粒度工作。
- 这会让盘中运行和收盘后运行在同一天共用缓存键，不够贴近真实交易流程。
- 因此本轮把 `advanced_ml` 链路统一改成：
  - `盘后生成信号`
  - `次日开盘执行`

### 本轮改动
1. 新增“最新已完成交易日”识别：
   - `daily_research/baseline/data_provider.py`
   - `get_latest_completed_trading_date()`
2. `advanced_ml` 训练/推理自动历史窗口不再默认取到“今天”，而是默认取到“最新已完成交易日”：
   - `daily_research/baseline/advanced_ml_runtime.py`
3. 训练目标从默认 `close->close` 扩展到 `next_open` 口径：
   - `daily_research/baseline/ml_alpha.py`
4. `run_advanced_daily_research.py` 改成显式用 `next_open` 口径训练和回测。
5. `generate_daily_trade_plan.py` 改成输出：
   - `信号日期`
   - `执行日期`
   - `盘后生成，次日开盘执行`
6. `train_trade_model.py` 导出的模型元信息新增：
   - `signal_date`
   - `execution_date`
   - `execution_mode=next_open`

### 验证
- 小样本训练验证通过：
  - `daily_research/execution/models/_nextopen_smoke.joblib`
  - `daily_research/execution/models/_nextopen_smoke.json`
- 小样本执行建议验证通过：
  - `daily_research/execution/output/nextopen_plan_smoke/daily_trade_plan.txt`
- 训练端在当前时间点自动识别：
  - `latest_data_date = 2026-03-19`
  - `execution_date = 2026-03-20`
  说明盘中不会把 `2026-03-20` 未收盘日线当成最终信号输入。

### 当前判断
- 这次改造是正确的。
- 它让执行端更贴近真实交易节奏，也让研究与执行的假设更加一致。
- 后续凡是要接近执行端的研究结果，都优先按这个 `盘后信号 -> 次日开盘执行` 口径验证。

## 2026-03-20 Deep Alpha 回测口径统一到 next_open

### 本轮改动
1. `deep_alpha` 目标构造改成支持 `next_open`
   - `daily_research/deep_alpha/sequence_dataset.py`
2. `run_deep_alpha_research.py` 默认把研究截止日期落到“最新已完成交易日”
3. `deep_alpha` holdout backtest 改成：
   - 盘后信号
   - 次日开盘执行
   - open-to-open 持有收益

### 烟测
- 运行：
  - `daily_research/output/deep_alpha_nextopen_smoke`
- 结果：
  - 超额收益：`0.42%`
  - 超额 Sharpe：`0.209`

### 当前判断
- `deep_alpha` 研究链路已经与执行端口径一致。
- 从现在开始，`deep_alpha` 与 `advanced_ml` 的对照，不再混用 close-to-close 与 next_open 两套假设。

## 2026-03-20 Next-Open 全A正面对照：Deep Alpha vs Advanced ML

### 目标
- 用同一套 `next_open` 口径，重新比较 `deep_alpha` 与 `advanced_ml`。
- 这一步不再看高流动性子样本，而是直接看全A范围内的同口径结果。
- 需要回答的问题很直接：
  1. `deep_alpha` 是否还能在更广泛的全A范围内保持优势。
  2. 如果不能，它距离执行端还有多远。

### 对照设置
- 统一口径：`盘后信号 -> 次日开盘执行`
- 股票池：全A
- 研究起点：`2024-01-01`
- `advanced_ml`
  - 输出目录：`daily_research/output/advanced_ml_alla_nextopen_compare_20260320`
  - holdout 切片：`daily_research/output/advanced_ml_alla_nextopen_compare_20260320/metrics_holdout_20250307_20260319.json`
- `deep_alpha`
  - 主线配置：
    - `raw return target`
    - `pairwise rank loss = 0.5`
    - `listwise rank loss = 0.25`
    - `state_gate`
  - 输出目录：`daily_research/output/deep_alpha_alla_raw_listwise25_e4_v252_nextopen_20260320`
- 汇总表：
  - `daily_research/output/deep_alpha_vs_advanced_ml_alla_nextopen_20260320.csv`

### 结果
#### Advanced ML
- 超额收益：`-12.51%`
- 超额 Sharpe：`-0.577`
- 超额最大回撤：`-24.77%`
- 平均持仓数：`3.82`
- 平均换手：`1.187`

#### Deep Alpha
- 超额收益：`-25.42%`
- 超额 Sharpe：`-0.434`
- 超额最大回撤：`-59.99%`
- 平均持仓数：`4.46`
- 平均换手：`1.168`

### 结论
- 这次全A `next_open` 正面对照的答案很明确：
  - `deep_alpha` 没有在全A范围内保持住此前在高流动性样本里的优势。
  - `advanced_ml` 虽然这一轮全A holdout 也没有跑赢基准，但整体仍然明显好于 `deep_alpha`。
- 因此当前不能把 `deep_alpha` 视为“开始接近执行端”的候选主线。
- 这次结果进一步确认：
  - `deep_alpha` 当前的有效性仍然明显依赖高流动性环境。
  - 它的泛化问题还没有解决。

### 当前决策
- `advanced_ml`
  - 继续保留为执行端默认主线。
- `deep_alpha`
  - 继续作为研究主线。
  - 当前不进入执行前最后验证阶段。

### 下一步收敛
- 不再优先讨论把 `deep_alpha` 接到执行端。
- 研究重点继续回到：
  1. 强化第一层 `return` 学习。
  2. 把流动性当条件变量，而不是普通特征。
  3. 优先解决“为什么只在高流动性环境里有效”的泛化问题。

## 2026-03-20 流动性条件化 return 学习：top_liquidity vs other

### 目标
- 不再把流动性做成独立 `return head`，而是把它收缩成两类条件：
  - `top_liquidity`
  - `other`
- 让流动性直接影响第一层 `return` 学习，而不是继续堆更复杂的结构：
  1. `return loss` 权重
  2. `ranking loss` 权重
  3. `sample weighting`

### 这次新增
- 已在 `deep_alpha` 训练器里加入：
  - `--liquidity-conditioning-mode top_vs_other`
  - `--top-liquidity-return-loss-weight`
  - `--top-liquidity-rank-loss-weight`
  - `--top-liquidity-sample-weight`
- 训练器现在会把最高流动性桶视为 `top_liquidity`，其余统一视为 `other`。
- 这一版只影响第一层 `return` 学习，不再切独立 head。

### 对照样本
- 高流动性样本：
  - `daily_research/output/deep_alpha_liquid500_20260319.txt`
- 分层全A子样本：
  - `daily_research/output/deep_alpha_stratified_alla800_20260320.txt`
- 汇总表：
  - `daily_research/output/deep_alpha_liquidity_conditioning_nextopen_compare_20260320.csv`

### 结果
#### liquid500
- `shared baseline`
  - 超额收益：`28.40%`
  - 超额 Sharpe：`1.250`
- `top_liquidity conditioning`（强）
  - 超额收益：`128.91%`
  - 超额 Sharpe：`4.639`
- `top_liquidity conditioning`（轻）
  - 超额收益：`310.10%`
  - 超额 Sharpe：`6.913`

#### stratified_all_a_800
- `shared baseline`
  - 超额收益：`-17.08%`
  - 超额 Sharpe：`-0.418`
- `top_liquidity conditioning`（强）
  - 超额收益：`-30.16%`
  - 超额 Sharpe：`-0.818`
- `top_liquidity conditioning`（轻）
  - 超额收益：`-12.39%`
  - 超额 Sharpe：`-0.333`

### 结论
- 这一步说明方向是有价值的：
  - 把流动性作为第一层 `return` 学习的条件变量，明显能放大 `liquid500` 上的 winner-picking 能力。
- 但它还没有完成我们真正想要的目标：
  - 在不牺牲高流动性优势的前提下，把分层全A泛化能力拉回来。
- 当前最好的观察是：
  - “轻条件化”比“强条件化”更稳。
  - 它在分层全A上有改善，但仍然没有转正。

### 当前判断
- 这条线值得继续研究，但还不能视为已解决泛化问题。
- 下一步不该再粗暴放大权重，而应该继续做：
  1. 更平滑的流动性条件化 loss
  2. 更细的高流动性/非高流动性状态诊断
  3. 让条件化只增强进攻，不破坏广样本稳定性

## 2026-03-20 更平滑的流动性条件化 loss + 结构拆解

### 目标
- 不再继续粗暴拉大 `top_liquidity` 的权重差，而是改成更平滑的条件化：
  - `smooth_bucket`
- 同时拆开回答两个更关键的问题：
  1. 高流动性里到底强化了哪些结构
  2. 非高流动性里到底是哪类结构在拖后腿

### 本次改动
- 训练器支持更平滑的流动性条件化：
  - `--liquidity-conditioning-mode smooth_bucket`
- 新增结构诊断脚本：
  - `daily_research/deep_alpha/analyze_liquidity_structures.py`

### 复验样本
- 高流动性样本：
  - `daily_research/output/deep_alpha_liquid500_20260319.txt`
- 分层全A子样本：
  - `daily_research/output/deep_alpha_stratified_alla800_20260320.txt`
- 汇总表：
  - `daily_research/output/deep_alpha_liquidity_conditioning_nextopen_compare_20260320.csv`

### 结果
#### liquid500
- `shared baseline`
  - 超额收益：`28.40%`
  - 超额 Sharpe：`1.250`
- `smooth_bucket`
  - 超额收益：`2.55%`
  - 超额 Sharpe：`0.056`
- 对照结论：
  - 更平滑不等于更强。
  - 这版 `smooth_bucket` 明显弱于此前的轻度 `top_liquidity conditioning`，也弱于 `shared baseline`。

#### stratified_all_a_800
- `shared baseline`
  - 超额收益：`-17.08%`
  - 超额 Sharpe：`-0.418`
- `smooth_bucket`
  - 超额收益：`-12.39%`
  - 超额 Sharpe：`-0.333`
- 对照结论：
  - `smooth_bucket` 没有转正，但和此前的“轻条件化”基本一致，说明：
    - 平滑条件化能稍微减少广样本退化
    - 但还不足以修复泛化

### 结构拆解
#### liquid500：高流动性里被强化的结构
- 诊断目录：
  - `daily_research/output/deep_alpha_liquid500_nextopen_smoothliq_20260320/liquidity_structure_analysis`
- 增强最明显的结构：
  1. `high_vol_expansion`
     - `rankic_delta = +0.1607`
     - `spread_delta = +0.1771`
  2. `trend_breakout`
     - `rankic_delta = +0.0295`
     - `spread_delta = +0.0257`
  3. `pullback_rebound`
     - `rankic_delta = +0.0154`
     - `spread_delta = +0.0063`
- 解释：
  - 条件化真正放大的，不是泛泛的“所有高流动性股票”，而是：
    - 高流动性中的突破推进
    - 高流动性中的高波动扩张
    - 高流动性中的回踩反弹

#### liquid500：非高流动性里拖后腿的结构
- 退化最明显的结构：
  1. `high_vol_expansion`
     - `rankic_delta = -0.1613`
     - `spread_delta = -0.0519`
  2. `low_vol_trend`
     - `rankic_delta = -0.0576`
  3. `weak_structure`
     - `rankic_delta = -0.0573`
  4. `neutral_mixed`
     - `spread_delta = -0.0285`
- 解释：
  - 这说明当前条件化会把“高流动性进攻结构”的偏好错误迁移到其他流动性层，导致：
    - 中低流动性里的高波动扩张被误判
    - 原本还能稳定工作的低波趋势、弱结构排序被破坏

#### stratified_all_a_800：更广样本里的结构信号
- 诊断目录：
  - `daily_research/output/deep_alpha_stratified_alla800_nextopen_smoothliq_20260320/liquidity_structure_analysis`
- 高流动性里仍然最有改善的是：
  - `trend_breakout`
    - `rankic_delta = +0.0400`
- 但“other”组里退化最明显的是：
  1. `neutral_mixed`
     - `rankic_delta = -0.4210`
  2. `pullback_rebound`
     - `rankic_delta = -0.3157`
  3. `trend_breakout`
     - `rankic_delta = -0.0790`
  4. `weak_structure`
     - `spread_delta = -0.0472`

### 当前结论
- “流动性作为条件变量”这条主线仍然成立。
- 但本次更平滑的 `smooth_bucket` 版本说明：
  - 单纯把条件化做平滑，并不会自动带来更强泛化
  - 真正被增强的是：
    - 高流动性里的 `trend_breakout`
    - 高流动性里的 `high_vol_expansion`
  - 真正被破坏的是：
    - 非高流动性里的 `neutral_mixed`
    - `pullback_rebound`
    - `low_vol_trend`
    - `weak_structure`

### 下一步收敛
- 继续把流动性当条件变量，但不再先调大权重。
- 下一步优先做：
  1. 只对“高流动性进攻结构”做定向增强，而不是整体抬升高流动性权重
  2. 对非高流动性里的 `neutral_mixed / pullback_rebound / low_vol_trend` 做保护性约束
  3. 让条件化更多地影响 `listwise / ranking`，而不是简单放大所有 `return loss`

## 2026-03-20 定向流动性结构增强：只强化进攻结构，保护 other 组基础排序

### 目标
- 不再继续粗暴放大高流动性整体权重。
- 改成更定向的结构条件化：
  - 只对 `top_liquidity` 下的
    - `trend_breakout`
    - `high_vol_expansion`
    做排序增强
  - 对 `other` 组里的
    - `neutral_mixed`
    - `pullback_rebound`
    - `low_vol_trend`
    做保护性排序加权
- 条件化优先作用在：
  - `pairwise rank loss`
  - `listwise loss`
- 尽量少碰全局 `return loss`

### 本次改动
- `daily_research/deep_alpha/sequence_dataset.py`
  - 把结构标签接进训练语料：
    - `trend_breakout`
    - `pullback_rebound`
    - `high_vol_expansion`
    - `low_vol_trend`
    - `weak_structure`
    - `neutral_mixed`
- `daily_research/deep_alpha/trainer.py`
  - 新增定向结构条件化排序权重
- `daily_research/deep_alpha/run_deep_alpha_research.py`
  - 新增参数：
    - `--structure-conditioning-mode targeted_liquidity_rank`
    - `--top-attack-structures`
    - `--other-protect-structures`
    - `--top-attack-rank-weight`
    - `--other-protect-rank-weight`

### 复验样本
- 汇总表：
  - `daily_research/output/deep_alpha_targeted_liquidity_structure_compare_20260320.csv`

### 结果
#### liquid500
- `shared baseline`
  - 超额收益：`28.40%`
  - 超额 Sharpe：`1.250`
- `top_liquidity conditioning`（轻）
  - 超额收益：`310.10%`
  - 超额 Sharpe：`6.913`
- `smooth_bucket`
  - 超额收益：`2.55%`
  - 超额 Sharpe：`0.056`
- `targeted_liquidity_rank`
  - 超额收益：`193.62%`
  - 超额 Sharpe：`5.309`
  - 超额最大回撤：`-16.32%`

#### stratified_all_a_800
- `shared baseline`
  - 超额收益：`-17.08%`
  - 超额 Sharpe：`-0.418`
- `top_liquidity conditioning`（轻）
  - 超额收益：`-12.39%`
  - 超额 Sharpe：`-0.333`
- `smooth_bucket`
  - 超额收益：`-12.39%`
  - 超额 Sharpe：`-0.333`
- `targeted_liquidity_rank`
  - 超额收益：`-10.31%`
  - 超额 Sharpe：`-0.251`
  - 总收益：`4.64%`

### 结构拆解
#### liquid500：定向增强真正强化了什么
- 诊断目录：
  - `daily_research/output/deep_alpha_liquid500_nextopen_targetedrank_20260320/liquidity_structure_analysis`
- `top_liquidity` 中提升最明显：
  1. `high_vol_expansion`
     - `rankic_delta = +0.2574`
     - `spread_delta = +0.1802`
  2. `pullback_rebound`
     - 变化很小，基本持平
  3. `trend_breakout`
     - 本次没有被强化，反而略退化
- 解释：
  - 当前定向增强真正学出来的是：
    - 高流动性里的高波动扩张进攻结构
  - 但 `trend_breakout` 这条线还没被稳定激活出来

#### liquid500：other 组里仍被拖后腿的结构
- 退化最明显：
  1. `neutral_mixed`
     - `rankic_delta = -0.0929`
  2. `low_vol_trend`
     - `rankic_delta = -0.0751`
  3. `high_vol_expansion`
     - `rankic_delta = -0.0610`
  4. `weak_structure`
     - `spread_delta = -0.0162`
- 解释：
  - `other` 组的保护还不够。
  - 尤其是：
    - `neutral_mixed`
    - `low_vol_trend`
  这两类结构仍然会被进攻型排序偏好破坏。

#### stratified_all_a_800：广样本里的变化
- 诊断目录：
  - `daily_research/output/deep_alpha_stratified_alla800_nextopen_targetedrank_20260320/liquidity_structure_analysis`
- `top_liquidity` 中改善最明显：
  1. `pullback_rebound`
     - `rankic_delta = +0.0902`
     - `spread_delta = +0.0412`
  2. `neutral_mixed`
     - `rankic_delta = +0.0431`
  3. `high_vol_expansion`
     - `rankic_delta = +0.0367`
- `other` 组中：
  - `neutral_mixed`
  - `weak_structure`
  - `trend_breakout`
  实际上也出现了改善
  - 但 `pullback_rebound` 仍明显退化

### 当前结论
- 这一步是有价值的，而且比 `smooth_bucket` 更接近我们想要的方向：
  - 不靠整体粗暴抬权重
  - 而是定向增强特定进攻结构
- 它带来的变化是：
  - `liquid500` 上继续显著增强
  - `stratified_all_a_800` 上也比 `shared / smooth_bucket / 轻条件化` 更好
- 但也要保持清醒：
  - 分层全A子样本仍然是负超额
  - 所以它还没有修复泛化问题，只是把问题从“整体失效”推进到了“更局部、可解释的失效”

### 下一步收敛
- 继续沿“定向结构增强”走，而不是回头做全局粗暴放大。
- 下一步最值得做：
  1. 继续想办法把 `trend_breakout` 在 `top_liquidity` 里真正强化出来
  2. 对 `other` 组里的 `neutral_mixed / low_vol_trend / pullback_rebound` 加更强保护
  3. 让保护优先作用在排序项，而不是全局 return loss

## 2026-03-20 Advanced ML：next_open 口径下的高流动性基础池对照

### 目标
- 因为执行规则已经统一成：
  - `盘后出策略`
  - `次日开盘执行`
- 所以前面基于旧交易假设得到的 `advanced_ml` 股票池结论，需要重新验证。
- 这次直接用 `next_open` 口径，对：
  - `liquid300`
  - `liquid500`
  - `liquid800`
  做正面对照，判断执行端最适合的高流动性基础池。

### 对照设置
- 模型：
  - `advanced_ml`
  - `HistGB`
  - 多周期 `5/10/20`
  - 周期权重 `0.2 / 0.3 / 0.5`
- 执行口径：
  - `next_open`
- 区间：
  - `2024-01-01` 起
- 输出汇总：
  - `daily_research/output/advanced_ml_liquidity_pool_compare_20260320.csv`

### 结果
#### liquid300
- 输出：
  - `daily_research/output/advanced_ml_liquid300_nextopen_20260320`
- 超额收益：`-0.76%`
- 超额 Sharpe：`-0.016`
- 超额最大回撤：`-37.04%`
- 结论：
  - 太窄了，edge 不稳，不适合作为执行端默认基础池。

#### liquid500
- 输出：
  - `daily_research/output/advanced_ml_liquid500_nextopen_20260320`
- 超额收益：`35.12%`
- 超额 Sharpe：`0.577`
- 超额最大回撤：`-26.54%`
- 结论：
  - 当前三组里，盈利能力最强。

#### liquid800
- 输出：
  - `daily_research/output/advanced_ml_liquid800_nextopen_20260320`
- 超额收益：`33.89%`
- 超额 Sharpe：`0.563`
- 超额最大回撤：`-25.49%`
- 结论：
  - 略逊于 `liquid500`，但更平衡一些。

### 当前判断
- 这次结论已经足够清楚：
  - `advanced_ml` 在 `next_open` 口径下仍然有效
  - 而且它在高流动性股票池里，明显好于此前更广的全A口径
- 三组里最适合执行端高流动性基础池的是：
  - **`liquid500`**
- 如果更偏保守、想要稍微更平衡一点的风格，可以把：
  - `liquid800`
  作为备选
- 但如果目标是“高盈利优先”，当前首选仍然是：
  - `liquid500`

### 含义
- 这一步也反过来说明：
  - 之前切到 `next_open` 之后，`advanced_ml` 在更广全A里表现变弱，不一定是模型完全失效
  - 更大的原因是：股票池过宽，把可交易 edge 稀释掉了
- 因此从执行角度看，后面不再追求“全A默认”，而改成：
  - `high-liquidity specialist universe`

### 下一步
- 执行端主线建议切换成：
  - `advanced_ml + liquid500`
- 研究端继续：
  - `deep_alpha` 在高流动性池里做更强进攻结构学习

## 2026-03-20 执行端正式切换到 liquid500 基础池
- 已给执行链路补齐 `--stocks-file` 支持：
  - `daily_research/baseline/train_trade_model.py`
  - `daily_research/baseline/generate_daily_trade_plan.py`
- 新增流动性股票池更新脚本：
  - `daily_research/execution/update_liquid_pool.py`
- 新增共享工具：
  - `daily_research/execution/liquidity_universe.py`
- 新流程：
  1. 盘后运行 `update_liquid_pool.py`
  2. 默认刷新 `liquid300 / liquid500 / liquid800`
  3. `update_model.py` 若未传 `--stocks` / `--stocks-file`，默认读 `liquid500_latest.txt`
  4. `run_trade_plan.py` 若未传 `--stocks` / `--stocks-file`，默认读 `liquid500_latest.txt`
- 已实际验证：
  - `update_liquid_pool.py --start-date 20240101` 成功写入 `daily_research/execution/universe/`
  - `run_trade_plan.py` 默认 liquid500 流程烟测通过：
    - `daily_research/execution/output/liquid500_default_smoke`
  - `update_model.py` 默认 liquid500 流程烟测通过：
    - `daily_research/execution/models/_liquid500_default_smoke.json`
- 当前判断：
  - 执行端主线：`advanced_ml + liquid500`
  - `deep_alpha` 继续作为高流动性专用研究主线推进

## 2026-03-20 Deep Alpha 固定到每日更新 liquid500 / liquid800 研究池
- 已在 `daily_research/deep_alpha/run_deep_alpha_research.py` 中新增 `--liquidity-pool`，支持直接读取：
  - `liquid300`
  - `liquid500`
  - `liquid800`
- 研究端现在可以直接复用执行端每日更新的固定股票池文件：
  - `daily_research/execution/universe/liquid500_latest.txt`
  - `daily_research/execution/universe/liquid800_latest.txt`
- 本轮继续用 `next_open` 口径、`Transformer + regression + pairwise/listwise + state_gate`，对每日更新的固定高流动性研究池做结构强化复验。

### 对照设置
- 基线：`shared`
- 结构强化：`targeted_liquidity_rank`
  - `top_attack_structures = trend_breakout, high_vol_expansion`
  - `other_protect_structures = neutral_mixed, pullback_rebound, low_vol_trend`
  - `top_attack_rank_weight = 1.2`
  - `other_protect_rank_weight = 1.08`
- 汇总：`daily_research/output/deep_alpha_fixed_liquidity_pool_compare_20260320.csv`

### liquid500（每日更新固定池）
- 基线输出：`daily_research/output/deep_alpha_liquid500_latest_shared_20260320`
- 强化输出：`daily_research/output/deep_alpha_liquid500_latest_targetedrank_20260320`
- `shared`
  - 超额收益：`70.13%`
  - 超额 Sharpe：`2.967`
  - 超额最大回撤：`-11.14%`
- `targetedrank`
  - 超额收益：`76.38%`
  - 超额 Sharpe：`2.302`
  - 超额最大回撤：`-22.41%`
- 结论：
  - 结构强化把收益推高了，但风险调整后收益和回撤明显变差。
  - 对当前每日更新的 `liquid500` 池来说，`shared` 仍然是更稳的高流动性研究基线。
- 结构诊断：`daily_research/output/deep_alpha_liquid500_latest_targetedrank_20260320/liquidity_structure_analysis/diagnosis.json`
  - 被明显伤到的仍然是：`high_vol_expansion`、`trend_breakout`
  - `other` 组里继续拖后腿的主要是：`neutral_mixed`、`low_vol_trend`、`weak_structure`

### liquid800（每日更新固定池）
- 基线输出：`daily_research/output/deep_alpha_liquid800_latest_shared_20260320`
- 强化输出：`daily_research/output/deep_alpha_liquid800_latest_targetedrank_20260320`
- `shared`
  - 超额收益：`64.92%`
  - 超额 Sharpe：`1.983`
  - 超额最大回撤：`-19.18%`
- `targetedrank`
  - 超额收益：`49.24%`
  - 超额 Sharpe：`1.199`
  - 超额最大回撤：`-29.01%`
- 结论：
  - 对当前每日更新的 `liquid800` 池，`targetedrank` 明显不如 `shared`。
- 结构诊断：`daily_research/output/deep_alpha_liquid800_latest_targetedrank_20260320/liquidity_structure_analysis/diagnosis.json`
  - 从结构指标上看，`trend_breakout / high_vol_expansion` 在 `top_liquidity` 内其实被强化了
  - 但组合层结果仍然变差，说明当前问题已经不只是“结构学没学到”，而是强化方式把风险/组合映射也一并扭坏了

### 当前判断
- `deep_alpha` 研究端现在正式固定为：
  - `liquid500`：高流动性主研究池
  - `liquid800`：更广的高流动性验证池
- 这轮结果说明：
  - 使用每日更新固定池是对的
  - 但当前这版结构强化（`targeted_liquidity_rank`）不够稳，暂不升级为新基线
- 后续继续沿“高流动性专用研究”推进，但默认参考基线改成：
  - `liquid500 shared`
  - `liquid800 shared`

## 2026-03-20 历史滚动高流动性股票池研究框架落地
- 目的：
  - 让研究端和当前交易模式彻底对齐
  - 不再用 `liquid500_latest.txt / liquid800_latest.txt` 静态回看历史
  - 正式解决高流动性研究里的成分前视/生存者偏差问题

### 本次新增
- `daily_research/execution/liquidity_universe.py`
  - 新增 `build_rolling_liquidity_membership(...)`
  - 能按历史 `ADV20` 排名构建：
    - `liquid300`
    - `liquid500`
    - `liquid800`
  - 默认每 `21` 个交易日重建一次
- `daily_research/deep_alpha/sequence_dataset.py`
  - 新增 `universe_membership_frame` 过滤
  - 训练样本只会在“当期滚动股票池成员”里生成
- `daily_research/deep_alpha/run_deep_alpha_research.py`
  - 新增：
    - `--rolling-liquidity-pool`
    - `--pool-rebalance-days`
    - `--pool-adv-window`
  - 输出：
    - `rolling_liquidity_schedule.csv`
    - `rolling_liquidity_summary.csv`
- `daily_research/baseline/run_advanced_daily_research.py`
  - 新增同口径的历史滚动高流动性研究入口
  - 可直接在 `advanced_ml` 上重做正式动态池验证

### 这一步的意义
- 执行端继续冻结：
  - `advanced_ml + liquid500 + next_open`
- 研究端主线继续保留：
  - `deep_alpha`
- 但从现在开始，凡是要讨论“是否接近执行端”的研究结论，都必须先通过：
  1. 历史滚动高流动性股票池
  2. `next_open`
  3. 足够长的样本外窗口

### 当前结论
- 这一步不是模型优化，而是研究地基校准。
- 从现在开始：
  - `liquid500_latest / liquid800_latest` 仍可用于**快速研究迭代**
  - **正式结论**优先看历史滚动高流动性框架

## 2026-03-20 正式框架重验：rolling liquid500 + next_open 下的 advanced_ml vs deep_alpha
- 目标：
  - 不再用静态 `liquid500_latest.txt` 回看历史
  - 改用历史滚动 `liquid500`
  - 用与当前执行端一致的 `next_open` 口径，重新正面对照：
    - `advanced_ml`
    - `deep_alpha`

### 实验设置
- 股票池：历史滚动 `liquid500`
- 重建频率：每 `21` 个交易日
- 研究起点：`2022-01-01`
- 对照窗口：`2025-03-07` 到 `2026-03-19`
- 输出：
  - `daily_research/output/advanced_ml_rolling_liq500_formal_20260320`
  - `daily_research/output/deep_alpha_rolling_liq500_formal_20260320`
  - `daily_research/output/deep_alpha_vs_advanced_ml_rolling_liq500_20260320.csv`

### advanced_ml（正式框架）
- 全样本输出：`daily_research/output/advanced_ml_rolling_liq500_formal_20260320`
- 动态历史池规模：
  - `rolling_pool_union_size = 2704`
  - `rolling_pool_rebalance_count = 49`
- 同口径 holdout 切片：
  - `window_start = 2025-03-07`
  - `window_end = 2026-03-19`
  - 超额收益：`12.44%`
  - 超额 Sharpe：`0.323`
  - 超额最大回撤：`-41.39%`

### deep_alpha（正式框架）
- 输出：`daily_research/output/deep_alpha_rolling_liq500_formal_20260320`
- 配置：
  - `Transformer`
  - `regression + pairwise/listwise`
  - `state_gate`
- 样本：
  - `train_samples = 208705`
  - `valid_samples = 107804`
- holdout：
  - `valid_start = 2025-03-07`
  - 超额收益：`30.12%`
  - 超额 Sharpe：`1.048`
  - 超额最大回撤：`-17.86%`

### 结论
- 在“历史滚动高流动性股票池 + next_open”这套正式框架下：
  - `deep_alpha` 重新建立了对 `advanced_ml` 的优势
  - 而且不是只赢收益，风险调整后收益和回撤也更优
- 这一步很关键，因为它说明：
  - 之前的静态池偏差确实会干扰判断
  - 但把框架校准正确后，`deep_alpha` 在高流动性专用模式下仍然有真实 edge

### 当前决策
- 执行端**暂不立即切换**
  - 继续保持：`advanced_ml + liquid500 + next_open`
- 但 `deep_alpha` 的状态升级为：
  - **重新进入执行前最后验证阶段候选**
- 接下来不再优先做宽泛模型微调，而是：
  1. 在同一正式框架下继续做更多 holdout / walk-forward 复验
  2. 如果优势继续保持，再进入 shadow mode 或执行候选阶段

## 2026-03-20 rolling liquid500 + next_open 多窗口 walk-forward 复验
### 目的
- 不再只看单一 holdout 的漂亮结果。
- 在同一正式框架下继续验证 `deep_alpha`：
  - 历史滚动 `liquid500`
  - `next_open`
  - 与当前执行主线 `advanced_ml` 做同池同口径对照
- 只有多窗口结果也稳定，才允许讨论 `shadow mode`。

### 对照窗口与输出
- 汇总：`daily_research/output/deep_alpha_rolling_liq500_walkforward_compare_20260320.csv`
- `deep_alpha`
  - `daily_research/output/deep_alpha_rolling_liq500_wf_20240822_20250305`
  - `daily_research/output/deep_alpha_rolling_liq500_wf_20250306_20250904`
  - `daily_research/output/deep_alpha_rolling_liq500_wf_20250905_20260319`
- `advanced_ml` 基准切片
  - `daily_research/output/advanced_ml_rolling_liq500_formal_20260320`

### 结果
#### 窗口 1：`2024-08-22 -> 2025-03-05`
- `advanced_ml`
  - 超额收益：`57.54%`
  - 超额 Sharpe：`2.489`
  - 超额最大回撤：`-24.42%`
- `deep_alpha`
  - 超额收益：`16.23%`
  - 超额 Sharpe：`1.912`
  - 超额最大回撤：`-7.91%`

#### 窗口 2：`2025-03-06 -> 2025-09-04`
- `advanced_ml`
  - 超额收益：`49.30%`
  - 超额 Sharpe：`3.026`
  - 超额最大回撤：`-21.16%`
- `deep_alpha`
  - 超额收益：`-9.65%`
  - 超额 Sharpe：`-0.692`
  - 超额最大回撤：`-14.62%`

#### 窗口 3：`2025-09-05 -> 2026-03-19`
- `advanced_ml`
  - 超额收益：`-28.78%`
  - 超额 Sharpe：`-1.382`
  - 超额最大回撤：`-41.39%`
- `deep_alpha`
  - 超额收益：`25.88%`
  - 超额 Sharpe：`2.889`
  - 超额最大回撤：`-8.66%`

### 结论
- `deep_alpha` 在正式框架下不是偶然强一次，它在最近窗口里确实表现出明显优势。
- 但它还没有形成“多数窗口稳定压过 `advanced_ml`”的形态。
- 尤其 `2025-03-06 -> 2025-09-04` 这一段，`deep_alpha` 明显失守，说明当前泛化还不够稳。

### 当前决策
- **暂不进入 `shadow mode`**
- 执行端继续冻结为：`advanced_ml + liquid500 + next_open`
- `deep_alpha` 继续保留为最有前景的研究主线，但下一阶段目标从“接近执行端”退回到“先把正式框架下的稳定性做出来”

### 下一步
1. 优先分析 `2025-03-06 -> 2025-09-04` 弱窗口里 `deep_alpha` 的失效结构。
2. 所有后续 `deep_alpha` 增强，必须继续在：
   - 历史滚动 `liquid500`
   - `next_open`
   - 多窗口 walk-forward
   这套正式框架下验证。
3. 只有多窗口稳定优于 `advanced_ml` 后，才重新讨论 `shadow mode`。

## 2026-03-21 trend_up_low_vol 定向排序增强复验
### 目标
- 只围绕 `trend_up_low_vol` 做第一层 return 学习增强。
- 优先修：
  - `neutral_mixed`
  - `pullback_rebound`
  - `trend_breakout`
- `low_vol_trend` 只做轻保护，不做激进强化。
- 所有新方案继续固定在：
  - 历史滚动 `liquid500`
  - `next_open`
  - 多窗口 walk-forward

### 本次实现
- 新增 `state_targeted_rank` 条件化模式：
  - 仅在目标状态下，对指定结构放大 pairwise/listwise 排序权重
  - 对 `low_vol_trend` 只做轻保护
- 关键代码：
  - `daily_research/deep_alpha/sequence_dataset.py`
  - `daily_research/deep_alpha/trainer.py`
  - `daily_research/deep_alpha/run_deep_alpha_research.py`
- 对照汇总：
  - `daily_research/output/deep_alpha_state_targeted_rank_compare_20260321.csv`

### 结果
#### 窗口 1：`2024-08-22 -> 2025-03-05`
- `deep_alpha shared`
  - 超额收益：`16.23%`
  - 超额 Sharpe：`1.912`
- `deep_alpha state_targeted_rank`
  - 超额收益：`-4.94%`
  - 超额 Sharpe：`-0.410`

#### 窗口 2：`2025-03-06 -> 2025-09-04`
- `deep_alpha shared`
  - 超额收益：`-9.65%`
  - 超额 Sharpe：`-0.692`
- `deep_alpha state_targeted_rank`
  - 超额收益：`-30.40%`
  - 超额 Sharpe：`-2.881`

#### 窗口 3：`2025-09-05 -> 2026-03-19`
- `deep_alpha shared`
  - 超额收益：`25.88%`
  - 超额 Sharpe：`2.889`
- `deep_alpha state_targeted_rank`
  - 超额收益：`7.31%`
  - 超额 Sharpe：`0.845`

### 结论
- 这条“直接在 loss 里做状态+结构定向放大”的路线，当前并不成立。
- 它不只没修复弱窗口，反而在三个窗口里都比 `shared` 更差。
- 这说明：
  - 当前问题不是“这些结构没有信息量”
  - 而是**直接对 ranking loss 做显式放大，会把组合映射带偏**

### 当前判断
- `deep_alpha` 继续保留为研究主线。
- 但后续不再优先尝试这种直接的 state-targeted loss 放大。
- 接下来更合理的方向应该是：
  1. 继续保留正式框架不变
  2. 不再直接扭 ranking loss 权重
  3. 转向更轻的结构表达/表示学习增强，而不是显式硬加权

## 2026-03-21 deep_alpha 弱窗口失效结构诊断
### 目的
- 解释 `2025-03-06 -> 2025-09-04` 这段 walk-forward 弱窗口里，为什么 `deep_alpha` 明显输给了 `advanced_ml`。
- 固化后续研究纪律：
  - 所有增强继续在
    - 历史滚动 `liquid500`
    - `next_open`
    - 多窗口 walk-forward
    下验证。

### 输出
- 诊断目录：`daily_research/output/deep_alpha_rolling_liq500_walkforward_failure_20260321`
- 关键文件：
  - `state_compare_summary.csv`
  - `structure_compare_summary.csv`
  - `state_structure_compare_summary.csv`
  - `liquidity_bucket_compare_summary.csv`
  - `diagnosis.json`

### 关键发现
#### 1. 弱窗口不是“全局都坏”，而是特定状态明显失守
- 弱窗口：`2025-03-06 -> 2025-09-04`
- 强窗口：`2025-09-05 -> 2026-03-19`
- 最关键差异在 `trend_up_low_vol`：
  - 弱窗口里，`deep_alpha` 的 `score_true20_rankic` 只有 `0.0165`
  - 强窗口里，同一状态升到 `0.1667`
  - 弱窗口里，这个状态的 top-bottom spread 还是负的：`-0.0273`
  - 强窗口里转正到：`0.0275`
- 这说明当前 `deep_alpha` 的主要问题不是“不会识别高流动性机会”，而是：
  - **在最重要的上涨低波状态里，排序稳定性不足**

#### 2. 弱窗口里真正拖后腿的结构
- 在 `trend_up_low_vol` 下，最明显的拖累结构是：
  - `neutral_mixed`
  - `pullback_rebound`
  - `trend_breakout`
- 这些结构在弱窗口里都出现了：
  - rankIC 低
  - spread 为负
- 而到了强窗口，同样几类结构都明显改善并转正。

#### 3. `low_vol_trend` 依然是持续性薄弱点
- 不管弱窗口还是强窗口，`low_vol_trend` 都偏弱。
- 弱窗口里更差：
  - 结构级 `score_true20_rankic = -0.0257`
- 强窗口里虽然回升，但也只到：
  - `0.0060`
- 这说明这类结构目前还不值得优先强化，反而更适合作为保护或降权对象。

#### 4. 流动性本身不是主矛盾，状态与结构的耦合才是
- 弱窗口和强窗口的主样本都集中在最高流动性桶：
  - 弱窗口 `bucket4` 样本 `43749`
  - 强窗口 `bucket4` 样本 `45455`
- 也就是说，当前失败并不是“流动性不够高”，而是：
  - **同样高流动性下，不同市场状态里的结构排序学得不够稳**

### 当前判断
- `deep_alpha` 仍然是最有前景的研究主线。
- 但当前最该修的不是：
  - 更换执行主线
  - 直接进入 `shadow mode`
- 而是：
  - 继续强化 `trend_up_low_vol` 下的 winner-picking
  - 优先修复 `neutral_mixed / pullback_rebound / trend_breakout`
  - 对 `low_vol_trend` 保持谨慎，先保护不放大

### 研究纪律
1. 执行端继续冻结为：`advanced_ml + liquid500 + next_open`
2. 后续所有 `deep_alpha` 增强都必须继续在：
   - 历史滚动 `liquid500`
   - `next_open`
   - 多窗口 walk-forward
   下验证
3. 只有多窗口稳定赢过 `advanced_ml`，才重新讨论 `shadow mode`

## 2026-03-21 执行端 / 研究端口径复核
### 结论
- 当前执行端和研究端不是“完全同频”，但这是**有意为之**，而且当前判断是合理的。
- 二者当前已经对齐的部分：
  - 都围绕高流动性股票池；
  - 都采用 `next_open` 交易口径；
  - 执行端冻结为 `advanced_ml + liquid500 + next_open`。
- 二者当前故意保留差异的部分：
  - 执行端流动池：`每日盘后更新 liquid500_latest.txt`；
  - 正式研究端流动池：`历史滚动 liquid500/liquid800`，默认每 `21` 个交易日重建一次。

### 这样设计的原因
- 执行端的目标是“明天开盘实际买什么”，所以应优先使用最新已完成交易日的流动性排名结果。
- 正式研究端的目标是“做可信验证”，所以必须避免拿今天的静态股票池回看历史，并降低过于频繁换池带来的噪声。

### 当前治理规则
1. `liquid500_latest.txt / liquid800_latest.txt`：用于执行端与快速研究诊断。
2. 历史滚动 `liquid500 / liquid800`：用于正式研究结论与策略晋级评估。
3. 没有在正式框架下通过验证的研究结果，不进入执行端，也不直接讨论 shadow mode。

## 2026-03-21 轻量结构表达增强：上下文嵌入复验
### 目标
- 保持正式框架不变：历史滚动 `liquid500`、`next_open`、多窗口 walk-forward。
- 不再直接扭 ranking loss 权重。
- 改为让模型更好表示结构信息：把 `state_id / liquidity_bucket / structure_id` 作为轻量上下文嵌入输入第一层 return 学习。

### 实现
- `MultiTaskRanker` 新增可选上下文嵌入：
  - `--state-context`
  - `--liquidity-context`
  - `--structure-context`
  - `--context-dim`
- 这些上下文不改变 loss 权重，只在编码器输出后做轻量表示融合。
- 本轮验证使用：`state + liquidity + structure` 三类上下文同时开启。

### 对照结果
- 汇总文件：`daily_research/output/deep_alpha_structure_context_compare_20260321.csv`

#### 窗口 1：`2024-08-22 -> 2025-03-05`
- `shared`
  - 超额收益：`16.23%`
  - 超额 Sharpe：`1.912`
- `context_repr`
  - 超额收益：`-10.99%`
  - 超额 Sharpe：`-0.977`

#### 窗口 2：`2025-03-06 -> 2025-09-04`
- `shared`
  - 超额收益：`-9.65%`
  - 超额 Sharpe：`-0.692`
- `context_repr`
  - 超额收益：`-2.67%`
  - 超额 Sharpe：`-0.338`

#### 窗口 3：`2025-09-05 -> 2026-03-19`
- `shared`
  - 超额收益：`25.88%`
  - 超额 Sharpe：`2.889`
- `context_repr`
  - 超额收益：`-11.66%`
  - 超额 Sharpe：`-1.297`

### 结论
- 这条“轻量结构表达增强”思路本身是对的，但当前这版上下文嵌入实现不成立。
- 它在弱窗口里有一定修复迹象，但明显破坏了两个本来表现较强的窗口。
- 也就是说：
  - 直接做 loss 级硬加权不对；
  - 当前这版把 `state/liquidity/structure` 一起作为上下文嵌入也不对。
- 下一步更合理的方向应当更克制：
  1. 不再一次性塞入三类上下文；
  2. 优先尝试更轻的结构表达，例如只做结构原型或结构平滑通道；
  3. 继续固定在正式框架下验证，没过多窗口就不讨论 `shadow mode`。

## 2026-03-22 辅助结构识别任务：正式框架复验
### 目标
- 在不改正式框架的前提下，尝试一条更轻的结构表达增强路线：
  - 不直接扭 `ranking loss` 权重；
  - 不再把 `state / liquidity / structure` 一起塞进上下文嵌入；
  - 改为增加一个**辅助结构识别任务**，让第一层表示更懂结构，再观察 return 排序是否改善。

### 实现
- `MultiTaskRanker` 新增轻量 `structure` 分类头。
- 新参数：
  - `--aux-structure-task`
  - `--aux-structure-loss-weight`
  - `--aux-structure-label-smoothing`
- 主任务仍然保持：
  - `raw return`
  - `pairwise + listwise`
  - `state_gate`
- 正式验证继续固定在：
  - 历史滚动 `liquid500`
  - `next_open`
  - 多窗口 walk-forward

### 输出
- 汇总文件：`daily_research/output/deep_alpha_aux_structure_walkforward_compare_20260322.csv`
- 三个窗口输出：
  - `daily_research/output/deep_alpha_rolling_liq500_auxstruct_wf_20240822_20250305`
  - `daily_research/output/deep_alpha_rolling_liq500_auxstruct_wf_20250306_20250904`
  - `daily_research/output/deep_alpha_rolling_liq500_auxstruct_wf_20250905_20260319`

### 结果
#### 窗口 1：`2024-08-22 -> 2025-03-05`
- `shared`
  - 超额收益：`16.23%`
  - 超额 Sharpe：`1.912`
- `aux_structure_task`
  - 超额收益：`4.21%`
  - 超额 Sharpe：`0.177`
  - 结构识别准确率：`74.50%`

#### 窗口 2：`2025-03-06 -> 2025-09-04`
- `shared`
  - 超额收益：`-9.65%`
  - 超额 Sharpe：`-0.692`
- `aux_structure_task`
  - 超额收益：`9.62%`
  - 超额 Sharpe：`0.558`
  - 结构识别准确率：`77.07%`

#### 窗口 3：`2025-09-05 -> 2026-03-19`
- `shared`
  - 超额收益：`25.88%`
  - 超额 Sharpe：`2.889`
- `aux_structure_task`
  - 超额收益：`-11.66%`
  - 超额 Sharpe：`-1.485`
  - 结构识别准确率：`83.16%`

### 结论
- 这条路线的信号很清楚：
  - 结构标签本身是**可学的**，因为辅助识别准确率并不低；
  - 但“学会识别结构”并不自动等于“更会做 return 排序”。
- 它确实修复了最弱窗口 `2025-03-06 -> 2025-09-04`；
- 但同时明显破坏了另外两个窗口，尤其是原本最强的 `2025-09-05 -> 2026-03-19`。
- 因此当前判断是：
  - **辅助结构识别任务不适合直接升级成 deep_alpha 主线配置**；
  - 但它证明了“结构表达增强”方向仍然成立，只是这版辅助任务与 return 排序的耦合方式不对。

### 后续约束
1. 执行端继续冻结为：`advanced_ml + liquid500 + next_open`
2. `deep_alpha` 继续保留为研究主线
3. 结构增强后续优先尝试：
   - 更单一的结构原型
   - 或结构平滑通道
4. 所有新方案继续固定在：历史滚动 `liquid500`、`next_open`、多窗口 walk-forward

## 2026-03-22 结构原型增强：正式框架复验
### 目标
- 继续保持正式框架不变：历史滚动 `liquid500`、`next_open`、多窗口 walk-forward。
- 不直接扭 `ranking loss` 权重，也不再加更多上下文信号。
- 改为让 embedding 围绕结构原型自然整理几何空间，观察能否更平稳地修复 return 排序。

### 实现
- `MultiTaskRanker` 新增可选 `structure_prototypes` 原型矩阵。
- 训练期增加轻量 prototype loss：
  - 用 embedding 与对应结构原型的相似度做约束；
  - 不直接改写主 return / ranking loss。
- 新参数：
  - `--structure-prototype-task`
  - `--structure-prototype-loss-weight`
  - `--structure-prototype-temperature`

### 输出
- 汇总文件：`daily_research/output/deep_alpha_structure_prototype_walkforward_compare_20260322.csv`
- 三个窗口输出：
  - `daily_research/output/deep_alpha_rolling_liq500_proto_wf_20240822_20250305`
  - `daily_research/output/deep_alpha_rolling_liq500_proto_wf_20250306_20250904`
  - `daily_research/output/deep_alpha_rolling_liq500_proto_wf_20250905_20260319`

### 结果
#### 窗口 1：`2024-08-22 -> 2025-03-05`
- `shared`
  - 超额收益：`16.23%`
  - 超额 Sharpe：`1.912`
- `structure_prototype`
  - 超额收益：`-9.01%`
  - 超额 Sharpe：`-0.338`
  - 原型识别准确率：`73.18%`

#### 窗口 2：`2025-03-06 -> 2025-09-04`
- `shared`
  - 超额收益：`-9.65%`
  - 超额 Sharpe：`-0.692`
- `structure_prototype`
  - 超额收益：`-4.23%`
  - 超额 Sharpe：`-0.286`
  - 原型识别准确率：`79.29%`

#### 窗口 3：`2025-09-05 -> 2026-03-19`
- `shared`
  - 超额收益：`25.88%`
  - 超额 Sharpe：`2.889`
- `structure_prototype`
  - 超额收益：`-7.98%`
  - 超额 Sharpe：`-1.084`
  - 原型识别准确率：`83.76%`

### 结论
- 结构原型同样证明了“结构可表示”这件事是成立的：原型识别准确率也不低。
- 但它和辅助结构识别任务有类似问题：
  - 确实对最弱窗口有小幅修复；
  - 但仍然破坏了两个本来更强的窗口。
- 因此当前判断是：
  - **结构原型增强也还不适合升级为 deep_alpha 主线配置**；
  - 它比直接硬加权更合理，但还没找到与 return ranking 稳定对齐的过渡方式。

### 后续约束
1. 执行端继续冻结为：`advanced_ml + liquid500 + next_open`
2. `deep_alpha` 继续保留为研究主线
3. 后续结构表达增强更应该走：
   - 更轻的结构平滑通道
   - 或更细的单结构局部表达
4. 不再继续堆叠“能识别结构但不能稳定提升排序”的辅助任务线

## 2026-03-22 Deep Alpha 表示学习主线升级
### 决策
- 下一阶段的大方向收敛为：
  - `patch-based masked self-supervised pretraining + return ranking fine-tune`
- 辅助任务、原型约束、本地 loss 重加权不再作为主研究路线。
- 正式评价框架保持不变：
  - rolling `liquid500`
  - `next_open`
  - multi-window walk-forward

### 已交付的新框架
- 新编码器族：
  - `patch_transformer`
- 新预训练模型：
  - `MaskedPatchPretrainer`
- 新入口脚本：
  - `daily_research/deep_alpha/pretrain_deep_alpha_encoder.py`
- fine-tune 入口新增支持：
  - `--encoder-family patch_transformer`
  - `--patch-len`
  - `--pretrained-encoder-path`

### 本阶段研究规则
1. 先在没有 return label 的条件下预训练编码器。
2. 再用同一编码器做 return ranking fine-tune。
3. 正式 A/B 只比较两组：
   - `shared baseline`
   - `masked-pretrained encoder + ranking fine-tune`
4. 如果它只修复弱窗口、却破坏强窗口，就停止推进。
5. 如果它没有在多窗口 walk-forward 中比 `shared` 更稳，也停止推进。

### 烟测验证
- 预训练烟测：
  - `daily_research/output/deep_alpha_pretrain_smoke_20260322`
- fine-tune 烟测：
  - `daily_research/output/deep_alpha_pretrained_finetune_smoke_20260322`
- 结论：
  - 两阶段流程已经可以端到端跑通；
  - 预训练编码器可以被 fine-tune 正常加载；
  - 下一步转入正式 A/B。

## 2026-03-22 正式 A/B：`shared` 基线 vs 掩码预训练编码器
### 正式框架
- rolling `liquid500`
- `next_open`
- multi-window walk-forward
- 只比较：
  - 当前 `shared baseline`
  - `masked-pretrained encoder + ranking fine-tune`

### 预训练运行
- `daily_research/output/deep_alpha_pretrain_liq500_formal_20240822_20250305`
- `daily_research/output/deep_alpha_pretrain_liq500_formal_20250306_20250904`
- `daily_research/output/deep_alpha_pretrain_liq500_formal_20250905_20260319`

### 微调运行
- `daily_research/output/deep_alpha_pretrained_liq500_formal_20240822_20250305`
- `daily_research/output/deep_alpha_pretrained_liq500_formal_20250306_20250904`
- `daily_research/output/deep_alpha_pretrained_liq500_formal_20250905_20260319`
- 汇总：
  - `daily_research/output/deep_alpha_pretrained_walkforward_compare_20260322.csv`

### 结果摘要
1. 窗口 `2024-08-22 -> 2025-03-05`
- `shared`: excess return `16.23%`, excess Sharpe `1.912`, excess MDD `-7.91%`
- `pretrained`: excess return `20.74%`, excess Sharpe `0.620`, excess MDD `-23.14%`
- 解读：
  - 收益略有提升，但稳定性明显变差。

2. 窗口 `2025-03-06 -> 2025-09-04`
- `shared`: excess return `-9.65%`, excess Sharpe `-0.692`, excess MDD `-14.62%`
- `pretrained`: excess return `-19.04%`, excess Sharpe `-0.993`, excess MDD `-27.34%`
- 解读：
  - 最弱窗口没有被修复，反而更差。

3. 窗口 `2025-09-05 -> 2026-03-19`
- `shared`: excess return `25.88%`, excess Sharpe `2.889`, excess MDD `-8.66%`
- `pretrained`: excess return `1.75%`, excess Sharpe `0.174`, excess MDD `-14.34%`
- 解读：
  - 原本最强的窗口被明显破坏。

### 结论
- 第一版大方向押注的 masked pretraining，并没有在正式框架下战胜当前 `shared` 基线。
- 它触发了停止规则：
  - 没有提升 walk-forward 稳定性；
  - 让弱窗口更差；
  - 明显破坏了最强窗口。
- 当前决策：
  - 执行端继续冻结为 `advanced_ml + liquid500 + next_open`
  - `deep_alpha` 继续保留为研究主线
  - 这版 masked-pretraining 配方先停止，不继续做局部打磨

## 2026-03-22 预训练轮数敏感性检查
### 问题
- 需要确认第一版 masked-pretraining 的失败，是否主要来自预训练预算过低。
- 本轮只改预训练 epoch 预算，其余配置保持不变。

### 正式设置
- rolling `liquid500`
- `next_open`
- fine-tune 配方保持不变
- 代表性窗口：
  - 弱窗口：`2025-03-06 -> 2025-09-04`
  - 强窗口：`2025-09-05 -> 2026-03-19`
- 汇总：
  - `daily_research/output/deep_alpha_pretrain_epoch_sensitivity_20260322.csv`

### 结果
#### 弱窗口 `2025-03-06 -> 2025-09-04`
- `shared baseline`: excess return `-9.65%`, excess Sharpe `-0.692`
- `pretrained 4e`: excess return `-19.04%`, excess Sharpe `-0.993`
- `pretrained 8e`: excess return `-3.39%`, excess Sharpe `-0.166`
- `pretrained 12e`: excess return `18.15%`, excess Sharpe `0.784`

#### 强窗口 `2025-09-05 -> 2026-03-19`
- `shared baseline`: excess return `25.88%`, excess Sharpe `2.889`
- `pretrained 4e`: excess return `1.75%`, excess Sharpe `0.174`
- `pretrained 8e`: excess return `-4.55%`, excess Sharpe `-0.540`
- `pretrained 12e`: excess return `21.24%`, excess Sharpe `1.876`

### 解读
- 预训练预算影响非常大。
- `4 epoch` 的失败不足以直接判死整条路线。
- `8 epoch` 依然不够。
- `12 epoch` 已经明显改变结论：
  - 修复了弱窗口；
  - 不再像早期版本那样彻底破坏强窗口；
  - 但仍未在强窗口上稳定战胜 `shared` 基线。

### 决策
- 这条 masked-pretraining 路线先不判死。
- 但也不能提前晋级。
- 下一步需要用 `12 epoch` 预训练，在完整多窗口上再做一次正式结论。

## 2026-03-22 训练预算护栏与本机运行策略
### 为什么需要这一步
- 第一版 `masked pretraining v1` 的结论被低预算预训练明显扭曲。
- 当时 `4 epoch` 结束时，预训练验证损失还在下降，而 fine-tune 已经接近平稳。
- 这说明问题不只在配方本身，也在于我们缺少识别 `undertraining` 的护栏。

### 代码改动
- 为 `deep_alpha` fine-tune 与 masked pretraining 同时加入训练诊断：
  - `best_epoch`
  - `best_valid_loss`
  - `final_valid_loss`
  - `stopped_early`
  - `still_improving`
  - `status`
  - `recommendation`
- 两个阶段都加入：
  - `ReduceLROnPlateau`
  - best-checkpoint restore
  - early stopping
- 为当前本机增加安全运行策略：
  - `patch_transformer` batch size 上限 `192`
  - Windows `num_workers` 上限 `2`
  - `prefetch_factor=1`
  - CUDA 可用时保留 `AMP`

### 当前决策
- 以后正式研究不能在不看训练诊断的情况下直接判死一条路线。
- 如果 `status = undertrained`，就不能给最终判决。
- 当前机器的推荐运行画像是：
  - 低 worker 数
  - 复用缓存
  - 中等 batch size
  - 开启 AMP

### 下一步
- 用 `12 epoch` 预训练重新跑完整正式结论。
- 正式框架保持不变：
  - rolling `liquid500`
  - `next_open`
  - multi-window walk-forward

## 2026-03-22 正式框架提速：`train_eval` 窗口裁剪
### 问题
- 正式 `deep_alpha` 流程变慢，不只是模型质量问题。
- 训练后的评估链条在当前本机上已经过重。
- 主要瓶颈是：
  - full-train `train_eval_loader`
  - then fitting `score_head`
  - then fitting `state_gate`

### 修正
- 正式框架保持不变：
  - rolling `liquid500`
  - `next_open`
  - multi-window walk-forward
- 但把 `train_eval_loader` 裁剪为最近一段 train-side 窗口，而不是整段训练区间。
- 新默认值：
  - `train_eval_window_days = 126`
- 这样既让校准更贴近近期市场，又显著降低正式运行成本。

### 代码改动
- `daily_research/deep_alpha/config.py`
- `daily_research/deep_alpha/run_deep_alpha_research.py`

### 新规则
- 不再默认使用 full-train 的 score-head / state-gate 拟合。
- 如需恢复旧行为，可显式加：
  - `--train-eval-window-days 0`

## 2026-03-22 更轻正式框架下的 12 轮预训练完整结论
### 为什么重跑
- 第一版 `12 epoch` 正式结论仍混杂了更重的后处理评估链条。
- 因此这次在更轻的正式默认值下重新跑完整 walk-forward：
  - rolling `liquid500`
  - `next_open`
  - `train_eval_window_days = 126`
  - explicit `num_workers = 0`

### 输出
- 汇总：
  - `daily_research/output/deep_alpha_pretrained_e12_lite126_walkforward_compare_20260322.csv`
- fine-tune 窗口：
  - `daily_research/output/deep_alpha_pretrained_liq500_formal_e12_lite126_wf_20240822_20250305`
  - `daily_research/output/deep_alpha_pretrained_liq500_formal_e12_lite126_wf_20250306_20250904`
  - `daily_research/output/deep_alpha_pretrained_liq500_formal_e12_lite126_wf_20250905_20260319`

### 结果
#### 窗口 `2024-08-22 -> 2025-03-05`
- `shared baseline`: excess return `16.23%`, excess Sharpe `1.912`, excess MDD `-7.91%`
- `pretrained e12 + lite126`: excess return `26.31%`, excess Sharpe `1.997`, excess MDD `-10.85%`

#### 窗口 `2025-03-06 -> 2025-09-04`
- `shared baseline`: excess return `-9.65%`, excess Sharpe `-0.692`
- `pretrained e12 + lite126`: excess return `-5.87%`, excess Sharpe `-0.478`, excess MDD `-16.60%`

#### 窗口 `2025-09-05 -> 2026-03-19`
- `shared baseline`: excess return `25.88%`, excess Sharpe `2.889`, excess MDD `-8.66%`
- `pretrained e12 + lite126`: excess return `14.41%`, excess Sharpe `1.566`, excess MDD `-6.16%`

### 训练诊断
- 预训练窗口 `2024-08-22 -> 2025-03-05`
  - `status = undertrained`
  - `best_epoch = 12/12`
- 预训练窗口 `2025-03-06 -> 2025-09-04`
  - `status = stable`
  - `best_epoch = 10/12`
- 预训练窗口 `2025-09-05 -> 2026-03-19`
  - `status = undertrained`
  - `best_epoch = 12/12`

### 解读
- 更轻的正式框架显著提升了本机研究吞吐，让完整结论可以稳定跑完。
- 在这套更干净的设置下，`12 epoch` masked pretraining 明显强于最早的 `4 epoch` 版本。
- 当前它已经：
  - 改善了第一个窗口的收益与 Sharpe；
  - 修复了部分弱窗口；
  - 但仍然没有保护住最强窗口的收益与 Sharpe。
- 更关键的是：
  - 窗口 `1` 与 `3` 的预训练结束时仍是 `undertrained`。

### 决策
- 按新的训练护栏，这条路线现在仍不能拿最终配方判决。
- 当前既不晋级执行，也不直接判死。
- 下一步：
  - 保持同一正式框架；
  - 把预训练预算上限从 `12 -> 16`；
  - 优先处理仍未平台化的窗口 `1` 与 `3`。

## 2026-03-22 自适应预训练预算支持
### 为什么新增
- 当前已经有足够证据表明：预训练预算会实质性改变研究结论。
- 因此不应该每次都凭经验猜一个固定上限。
- 更合理的方式是：
  - 当诊断仍显示 `undertrained` 时，允许在同一轮运行里自动续训；
  - 同时保留明确的硬上限。

### 改动
- `daily_research/deep_alpha/trainer.py`
- `daily_research/deep_alpha/pretrain_deep_alpha_encoder.py`

新增预训练控制参数：
- `--auto-extend-undertrained`
- `--epoch-extend-step`
- `--max-total-epochs`

行为：
- 预训练先从指定的 `--epochs` 启动；
- 如果跑到上限后诊断仍为 `undertrained`，预算会按 `epoch_extend_step` 自动上调；
- 到达 `max_total_epochs` 后停止；
- 整个过程保持同一轮 optimizer / scheduler 状态，不再为每次预算测试重新起跑。

### 当前政策
- 自适应续训不是“自动放行”。
- 最终研究结论仍然要求：
  - 关键窗口不再是 `undertrained`
  - 正式框架下通过多窗口 walk-forward
- 对当前本机，安全运行策略仍保持克制：
  - 如果显式指定 `num_workers = 0`，就尊重它；
  - batch size 仍然受安全上限约束。

## 2026-03-22 项目治理与维护整理
### 项目总评
- `daily_research` 已经形成清晰分层：
  - `baseline` 负责可解释基线与 advanced ML 主线；
  - `execution` 负责盘后更新与次日开盘执行；
  - `deep_alpha` 负责表示学习研究，不直接进入执行端。
- 当前真正冻结可执行主线仍然是：
  - `advanced_ml + liquid500 + next_open`
- `t0_project` 应继续视为独立实验区，不与 `daily_research` 的正式执行链路混用。

### 本轮修正的历史问题
- 修复了执行入口长期积累的重复启动逻辑：
  - `daily_research/execution/update_model.py`
  - `daily_research/execution/run_trade_plan.py`
  - 已统一抽到 `daily_research/execution/entrypoint_utils.py`
- 修复了 `deep_alpha` 对主研究脚本内部私有函数的耦合：
  - 公共数据装载、股票池解析、切分日期、滚动池缓存等流程已抽到 `daily_research/deep_alpha/pipeline_utils.py`
  - `pretrain_deep_alpha_encoder.py` 不再依赖 `run_deep_alpha_research.py` 的内部下划线函数
- 新增工作区维护工具：
  - `daily_research/tools/workspace_maintenance.py`
  - 用于体检目录体积、识别生成物并安全清理可再生产物
- 新增顶层维护说明：
  - 工作区根目录 `README.md`
  - 工作区根目录 `.gitignore`

### 当前维护结论
- 代码层面：
  - 本轮整理后，核心 Python 脚本已通过编译检查；
  - 入口重复、跨文件私有依赖、工作区边界不清这三类历史问题已经被压缩。
- 产物层面：
  - 当前主要空间占用仍来自 `daily_research/cache/` 与 `daily_research/output/`；
  - 这些目录属于研究产物，不应作为日常默认清理目标。
- 维护纪律：
1. 默认只清理 `__pycache__`、`*.pyc`、日志、TensorBoard 事件文件等可再生产物。
2. 研究结论继续统一沉淀到本日志；工作区级规则维护在根目录 `README.md`。
3. `deep_alpha` 后续若未通过正式框架，不讨论执行接入。

## 2026-03-22 Git 纳管与归档规则建立
### 本轮治理目标
- 把当前工作区正式纳入 Git 管理。
- 把 `daily_research/cache/` 与 `daily_research/output/` 从“持续堆积区”改成“热区 + 冷归档”结构。

### 新规则
- Git 跟踪范围：
  - 源码
  - 文档
  - `daily_research/archive_policy.json`
  - `daily_research/archive/manifests/` 下的小型归档清单
- Git 明确不跟踪：
  - `daily_research/output/`
  - `daily_research/cache/`
  - `daily_research/archive/cache/`
  - `daily_research/archive/output/`
  - 以及其他生成物目录

### 归档策略
- `output/`：
  - 最近 `21` 天或最近 `12` 个实验目录保留在热区；
  - 更旧的实验目录进入 `daily_research/archive/output/`。
- `output/` 根目录下的汇总表、日志、说明文件：
  - 最近 `30` 天或最近 `30` 个文件保留在热区；
  - 更旧文件进入 `daily_research/archive/output/root_files/`。
- `cache/`：
  - 对大体积哈希缓存按桶管理：
    - `deep_alpha/raw`
    - `deep_alpha/features`
    - `deep_alpha/corpus`
    - `advanced_ml/raw`
    - `advanced_ml/prepared`
  - 这些桶按“最近保留数量 + 最近保留天数”双阈值决定是否进入冷归档。
- 默认不归档的小型常驻缓存：
  - `industry_map_tq.csv`
  - `style_map_tq.csv`
  - `rolling_pools`
  - `states`
  - `liquidity_buckets`

### 工具化落地
- 新增归档策略文件：
  - `daily_research/archive_policy.json`
- 新增归档说明目录：
  - `daily_research/archive/README.md`
- `workspace_maintenance.py` 新增 `archive` 子命令：
  - dry-run：先列出候选；
  - `--apply`：再执行移动；
  - 执行后自动生成 manifest。

### 维护纪律
1. 任何归档先 dry-run，再 `--apply`。
2. 归档后优先提交 manifest 与规则文件，不提交 payload 本体。
3. 若某个缓存桶将来出现误判，再调规则，不直接恢复为“长期全部热存”。

## 2026-03-22 执行端收益能力体检与研究重启
### 体检依据
- 正式执行主线目录：
  - `daily_research/output/advanced_ml_rolling_liq500_formal_20260320`
- 股票池对照：
  - `daily_research/output/advanced_ml_liquidity_pool_compare_20260320.csv`
- 体检工具：
  - `daily_research/tools/execution_health_check.py`

### 总体结果
- 在正式框架 `rolling liquid500 + next_open` 下，截至 `2026-03-19`：
  - 总超额收益：`224.21%`
  - 总超额 Sharpe：`1.037`
  - 总超额最大回撤：`-41.39%`
- 这说明长期样本下，当前执行主线仍然是有效的。

### 近期结果
- 最近完整 holdout `2025-03-07 -> 2026-03-19`：
  - 超额收益：`12.44%`
  - 超额 Sharpe：`0.323`
  - 超额最大回撤：`-41.39%`
- 最新子窗口 `2025-09-05 -> 2026-03-19`：
  - 超额收益：`-28.78%`
  - 超额 Sharpe：`-1.382`
  - 超额最大回撤：`-41.39%`

### 股票池判断
- `liquid300 / liquid500 / liquid800` 对照结果显示：
  - `liquid500` 的超额 Sharpe 仍是三档高流动性池里最高；
  - `liquid800` 接近，但没有稳定优于 `liquid500`。
- 因此这轮不先改执行股票池，继续固定为 `liquid500`。

### 结论
- 当前执行主线不是“整体失效”，而是“长期仍有效，但近期收益能力已经不理想”。
- 真正触发重启研究的核心证据是：
  - 最新正式子窗口已经出现负超额收益；
  - 最新正式子窗口超额 Sharpe 为负；
  - 整体超额回撤仍然偏深。

### 当前决策
1. 执行端默认值继续冻结为：`advanced_ml + liquid500 + next_open`
2. 不直接切换到 `deep_alpha`，因为它还没有通过完整晋级标准
3. 从现在开始，正式重启执行方向研究

### 重启范围
1. 固定正式框架不变：
   - 历史滚动 `liquid500`
   - `next_open`
   - 多窗口 walk-forward
2. 第一轮优先研究：
   - `advanced_ml` 集成权重
   - 状态启停
   - 回撤控制
   - 换手约束
3. 研究目标不是先扩新路线，而是先修复当前执行主线在 `2025-09-05 -> 2026-03-19` 的弱窗口表现

## 2026-03-22 执行端弱窗口第一轮正式修复扫描
### 本轮目的
- 不改执行默认值，也不先改股票池。
- 只在正式框架下检查：当前弱窗口能否通过执行线自身的小修复得到明显改善。
- 第一轮优先看两类杠杆：
  - `trend_up_low_vol` 下的分段集成权重；
  - 换手约束。

### 为跑正式扫描补的历史兼容修复
- 在 `quant` 环境下，项目里一批旧脚本会因为 `Python 3.9` 的类型注解兼容问题直接报错。
- 本轮顺手补齐了这些文件的 `from __future__ import annotations`：
  - `daily_research/baseline/alpha.py`
  - `daily_research/baseline/backtest.py`
  - `daily_research/baseline/data_provider.py`
  - `daily_research/baseline/portfolio.py`
  - `daily_research/baseline/position_manager.py`
  - `daily_research/baseline/run_daily_research.py`
- 这一步不改变策略逻辑，只是清理旧环境兼容债，确保正式研究脚本在现有研究环境里可直接运行。

### 新增工具
- `daily_research/baseline/scan_execution_repair_candidates.py`
- 作用：
  - 固定正式口径 `rolling liquid500 + next_open`
  - 共享同一份原始数据、因子准备和滚动 ML 分数
  - 批量比较执行线修复候选
  - 自动输出全样本、最近完整窗口和最新弱窗口摘要

### 正式扫描输出
- 目录：
  - `daily_research/output/advanced_ml_execution_repair_scan_20260322_formal_round1`
- 汇总：
  - `daily_research/output/advanced_ml_execution_repair_scan_20260322_formal_round1/repair_scan_summary.csv`
- 基线对候选归因：
  - `daily_research/output/advanced_ml_execution_repair_scan_20260322_formal_round1/attr_baseline_vs_up_low_ml55_none25_v220`

### 候选结果
#### 基线
- `baseline`
  - 全样本超额收益：`223.44%`
  - 全样本超额 Sharpe：`1.035`
  - 最新弱窗口超额收益：`-28.95%`
  - 最新弱窗口超额 Sharpe：`-1.381`
  - 最新弱窗口平均换手：`0.940`

#### 候选 1：只调低 `trend_up_low_vol` 的 ML 主导权重
- `up_low_ml55_none25_v220`
  - 状态权重：`trend_up_low_vol -> ml=0.55 / none=0.25 / v2=0.20`
  - 全样本超额收益：`200.21%`
  - 全样本超额 Sharpe：`1.035`
  - 全样本超额最大回撤：`-31.22%`
  - 最近完整窗口超额收益：`33.11%`
  - 最近完整窗口超额 Sharpe：`0.898`
  - 最新弱窗口超额收益：`-10.76%`
  - 最新弱窗口超额 Sharpe：`-0.582`
  - 最新弱窗口平均换手：`1.071`

#### 候选 2：更激进地下调 `trend_up_low_vol` 的 ML 权重
- `up_low_ml50_none20_v230`
  - 状态权重：`trend_up_low_vol -> ml=0.50 / none=0.20 / v2=0.30`
  - 全样本超额收益：`151.50%`
  - 全样本超额 Sharpe：`0.875`
  - 最新弱窗口超额收益：`-7.84%`
  - 最新弱窗口超额 Sharpe：`-0.466`
- 结论：
  - 弱窗口修得更狠；
  - 但全样本收益和 Sharpe 损失更大，不是当前最稳候选。

#### 候选 3：单独收紧换手约束
- `turnover1_hold3`
  - 约束：`turnover_limit=1.0`，`min_hold_days=3`
  - 全样本超额收益：`141.97%`
  - 全样本超额 Sharpe：`0.833`
  - 最新弱窗口超额收益：`-29.10%`
  - 最新弱窗口超额 Sharpe：`-1.550`
- 结论：
  - 单独收紧换手约束没有修复弱窗口；
  - 反而同时伤害了全样本和近期窗口。

#### 候选 4：状态权重修复 + 更严换手约束
- `up_low_ml55_none25_v220_turnover1_hold3`
- `up_low_ml50_none20_v230_turnover1_hold3`
- 结论：
  - 两者都明显差于只做状态权重修复；
  - 当前不应把更严换手约束作为第一优先修复方向。

### 弱窗口结构判断
- 最新弱窗口的拖累主要集中在：
  - `trend_up_low_vol`
  - 次要是 `trend_down_low_vol`
- 基线在弱窗口里：
  - `trend_up_low_vol` 超额收益约 `-29.68%`
  - `trend_down_low_vol` 超额收益约 `-8.14%`
- `up_low_ml55_none25_v220` 修复后：
  - `trend_up_low_vol` 超额收益改善到约 `-11.58%`
  - `trend_down_low_vol` 超额收益改善到约 `-1.05%`
- 这说明当前最有效的修复不是“少交易”，而是“减少 `trend_up_low_vol` 状态下 ML 分数的主导性”。

### 归因补充
- 基线与 `up_low_ml55_none25_v220` 的全样本归因显示：
  - `repair` 相对基线最强季度是 `2025Q4`
  - 超额差值约 `+8.82%`
- 但它在更早的强窗口有一定让利，因此当前仍只能算“最稳修复候选”，不能直接升级成新默认值。

### 本轮结论
1. 第一轮正式修复扫描已经确认：执行线当前最值得继续做的是 `trend_up_low_vol` 分段权重修复，而不是先收紧换手。
2. 当前最稳候选是：
   - `trend_up_low_vol -> ml=0.55 / none=0.25 / v2=0.20`
3. 这个候选已经显著改善了最新弱窗口，同时保住了全样本超额 Sharpe，并明显收敛了全样本超额回撤。
4. 但它仍没有把弱窗口修回正收益，所以执行端默认值继续冻结，不直接切换。

### 下一步
1. 继续围绕 `trend_up_low_vol` 分段权重做更细的正式扫描。
2. 把第二轮重点放在：
   - `0.55 ~ 0.60` 一带的分段权重微调
   - 状态启停阈值
3. 暂不把更严换手约束作为第一优先修复方向。

## 2026-03-22 执行端弱窗口第二轮正式扫描
### 本轮目标
- 延续第一轮正式修复研究，但把范围进一步收敛。
- 不再继续优先扫描更严的换手约束。
- 第二轮只做两件事：
  - 继续微调 `trend_up_low_vol` 下的分段集成权重；
  - 正式检查“状态启停阈值”是否比单纯权重微调更有效。

### 工具补充
- `daily_research/baseline/scan_execution_repair_candidates.py`
  - 新增 `--candidate-set`
  - 当前支持：
    - `round1`
    - `round2_weights`
    - `pair_best`
- 这样第二轮可以把“权重扫描”和“状态阈值扫描”拆开跑，避免变量混在一起。

### 扫描输出
- 权重微调：
  - `daily_research/output/advanced_ml_execution_repair_scan_20260322_formal_round2_weights`
- 状态启停阈值：
  - `daily_research/output/advanced_ml_execution_repair_scan_20260322_formal_round2_ma50`
  - `daily_research/output/advanced_ml_execution_repair_scan_20260322_formal_round2_ma55`
  - `daily_research/output/advanced_ml_execution_repair_scan_20260322_formal_round2_ma40`
- `ma60` 基线对 `ma50` 基线归因：
  - `daily_research/output/advanced_ml_execution_repair_scan_20260322_formal_round2_ma50/attr_ma60_baseline_vs_ma50_baseline`

### 结果一：`ma60` 框架下的权重微调
#### 结果总览
- `baseline`
  - 全样本超额 Sharpe：`1.035`
  - 最新弱窗口超额收益：`-28.95%`
  - 最新弱窗口超额 Sharpe：`-1.381`
- `up_low_ml60_none25_v215`
  - 全样本超额 Sharpe：`0.974`
  - 最新弱窗口超额收益：`-15.53%`
  - 最新弱窗口超额 Sharpe：`-0.858`
- `up_low_ml58_none24_v218`
  - 全样本超额 Sharpe：`0.819`
  - 最新弱窗口超额收益：`-18.33%`
  - 最新弱窗口超额 Sharpe：`-0.996`
- `up_low_ml57_none25_v218`
  - 全样本超额 Sharpe：`0.913`
  - 最新弱窗口超额收益：`-17.35%`
  - 最新弱窗口超额 Sharpe：`-0.921`
- `up_low_ml56_none24_v220`
  - 全样本超额 Sharpe：`0.928`
  - 最新弱窗口超额收益：`-15.61%`
  - 最新弱窗口超额 Sharpe：`-0.858`
- `up_low_ml55_none25_v220`
  - 全样本超额 Sharpe：`1.035`
  - 最新弱窗口超额收益：`-10.76%`
  - 最新弱窗口超额 Sharpe：`-0.582`

#### 结论
- 在 `0.55 ~ 0.60` 区间里，第一轮的最优点 `0.55 / 0.25 / 0.20` 没有被推翻。
- 其余更靠近 `0.60` 的方案虽然也能改善弱窗口，但都不如 `0.55 / 0.25 / 0.20` 稳。
- 因此：
  - `ma60` 框架下的权重修复已经基本收敛；
  - 继续围绕这条线盲扫，边际收益已经不高。

### 结果二：状态启停阈值扫描
#### `regime_ma_window=50`
- `baseline`
  - 全样本超额收益：`580.31%`
  - 全样本超额 Sharpe：`1.841`
  - 全样本超额最大回撤：`-29.11%`
  - 最近完整窗口超额收益：`77.33%`
  - 最近完整窗口超额 Sharpe：`1.896`
  - 最新弱窗口超额收益：`-10.87%`
  - 最新弱窗口超额 Sharpe：`-0.559`
- `up_low_ml55_none25_v220`
  - 弱窗口略进一步改善到 `-9.31% / -0.509`
  - 但全样本明显不如 `ma50 baseline`

#### `regime_ma_window=55`
- `baseline`
  - 全样本超额 Sharpe：`0.549`
  - 最新弱窗口超额收益：`-23.21%`
  - 最新弱窗口超额 Sharpe：`-1.076`
- `up_low_ml55_none25_v220`
  - 全样本超额收益：`207.15%`
  - 全样本超额 Sharpe：`1.057`
  - 最新弱窗口超额收益：`-9.33%`
  - 最新弱窗口超额 Sharpe：`-0.471`

#### `regime_ma_window=40`
- `baseline`
  - 全样本超额 Sharpe：`1.121`
  - 最新弱窗口超额收益：`-20.17%`
  - 最新弱窗口超额 Sharpe：`-1.048`
- `up_low_ml55_none25_v220`
  - 全样本超额 Sharpe：`0.802`
  - 最新弱窗口超额收益：`-21.63%`
  - 最新弱窗口超额 Sharpe：`-1.194`

#### 结论
- `ma40` 明显偏紧，不是当前方向。
- `ma55` 只有在叠加 `0.55 / 0.25 / 0.20` 时才恢复到可用，但整体仍不如 `ma50 baseline`。
- 当前最强的新信号不是“继续压 `trend_up_low_vol` 权重”，而是“把状态启停从 `ma60` 收到 `ma50`”。

### 三窗口交叉核对
- `ma60 baseline`
  - 窗口 1：`57.54% / 2.489`
  - 窗口 2：`49.30% / 3.006`
  - 窗口 3：`-28.95% / -1.381`
- `ma60 + up_low_ml55_none25_v220`
  - 窗口 1：`47.13% / 2.161`
  - 窗口 2：`45.66% / 2.920`
  - 窗口 3：`-10.76% / -0.582`
- `ma55 + up_low_ml55_none25_v220`
  - 窗口 1：`9.75% / 0.413`
  - 窗口 2：`84.93% / 5.702`
  - 窗口 3：`-9.33% / -0.471`
- `ma50 baseline`
  - 窗口 1：`85.94% / 4.180`
  - 窗口 2：`93.95% / 6.304`
  - 窗口 3：`-10.87% / -0.559`

### 归因补充
- `ma50 baseline` 相对当前 `ma60 baseline`：
  - 17 个季度里有 13 个季度超额更强
  - 最强季度是 `2026Q1`
  - 超额差约 `+19.23%`
- 市场状态归因：
  - `trend_up_low_vol` 超额差约 `+145.98%`
  - `trend_down_low_vol` 超额差约 `+36.48%`
- 这说明：
  - `ma50` 不是只修了一个小弱窗口；
  - 它在正式框架下对主要有效状态的映射更强。

### 本轮结论
1. `ma60` 框架下的权重微调已经收敛，`0.55 / 0.25 / 0.20` 仍是这条线最优点。
2. 第二轮更关键的新发现是：`regime_ma_window=50` 的 `baseline` 已经在三个正式窗口里同时优于当前执行主线。
3. 因此执行端修复研究的第一优先级应当更新为：
   - 先复验并确认 `ma50` 状态启停候选；
   - 再决定是否还需要在 `ma50` 框架里叠加 `trend_up_low_vol` 权重微调。
4. 更严换手约束继续不列为第一优先修复方向。

### 当前决策
1. 执行端默认值暂不切换，继续冻结为：`advanced_ml + liquid500 + next_open`
2. 但新的“最强正式修复候选”已经从第一轮的 `ma60 + up_low_ml55_none25_v220`，更新为第二轮的 `ma50 baseline`
3. `ma60 + up_low_ml55_none25_v220` 仍保留为次一级备选

## 2026-03-22 执行端弱窗口第三轮正式复验：`ma50 baseline`
### 本轮目标
- 不再继续盲扫新的大分支，而是专门确认第二轮跑出来的 `ma50 baseline` 是否真的站得住。
- 本轮只回答三件事：
  - `ma50 baseline` 是否继续优于当前 `ma60 baseline`
  - `ma50 baseline` 是否继续优于次一级备选 `ma60 + up_low_ml55_none25_v220`
  - 在 `ma50` 框架里，是否还值得立刻进入第四轮权重微调

### 本轮产物
- `daily_research/output/advanced_ml_execution_repair_scan_20260322_formal_round3_ma60_pair`
- `daily_research/output/advanced_ml_execution_repair_scan_20260322_formal_round3_ma50_pair`
- `daily_research/output/advanced_ml_execution_repair_scan_20260322_formal_round3_ma50_revalidation`
- 关键归因目录：
  - `daily_research/output/advanced_ml_execution_repair_scan_20260322_formal_round3_ma50_revalidation/attr_ma60_baseline_vs_ma50_baseline`
  - `daily_research/output/advanced_ml_execution_repair_scan_20260322_formal_round3_ma50_revalidation/attr_ma60_up_low_vs_ma50_baseline`
  - `daily_research/output/advanced_ml_execution_repair_scan_20260322_formal_round3_ma50_revalidation/attr_ma50_baseline_vs_ma50_up_low`

### 结果一：`ma50 baseline` 对当前执行主线
- `ma60 baseline`
  - 全样本超额收益：`223.44%`
  - 全样本超额 Sharpe：`1.035`
  - 全样本超额最大回撤：`-41.53%`
  - 最近完整窗口：`12.17% / 0.315`
  - 最新弱窗口：`-28.95% / -1.381`
- `ma50 baseline`
  - 全样本超额收益：`580.31%`
  - 全样本超额 Sharpe：`1.841`
  - 全样本超额最大回撤：`-29.11%`
  - 最近完整窗口：`77.33% / 1.896`
  - 最新弱窗口：`-10.87% / -0.559`
- 季度稳定性：
  - `ma50 baseline` 在 `17` 个季度里有 `13` 个季度超额更强
  - 最强季度是 `2026Q1`
  - 单一最强季度对正向季度总优势的占比约 `20.96%`
- 这说明：
  - `ma50 baseline` 不是只靠某一个季度抬起来；
  - 它对当前执行主线的优势是跨季度、跨窗口的。

### 结果二：`ma50 baseline` 对次一级备选
- `ma60 + up_low_ml55_none25_v220`
  - 全样本超额收益：`200.21%`
  - 全样本超额 Sharpe：`1.035`
  - 全样本超额最大回撤：`-31.22%`
  - 最近完整窗口：`33.11% / 0.898`
  - 最新弱窗口：`-10.76% / -0.582`
- `ma50 baseline`
  - 全样本超额收益：`580.31%`
  - 全样本超额 Sharpe：`1.841`
  - 全样本超额最大回撤：`-29.11%`
  - 最近完整窗口：`77.33% / 1.896`
  - 最新弱窗口：`-10.87% / -0.559`
- 季度稳定性：
  - `ma50 baseline` 同样是在 `17` 个季度里赢下 `13` 个季度
  - 最强季度是 `2025Q2`
  - 单一最强季度对正向季度总优势的占比约 `19.84%`
- 这说明：
  - 即使把第一轮最稳的 `ma60` 权重修补拿来对照，`ma50 baseline` 仍然是更强的正式候选；
  - 它的优势同样不是单季度异常造成。

### 结果三：`ma50` 框架内部是否还要立刻做权重微调
- `ma50 baseline`
  - 全样本超额 Sharpe：`1.841`
  - 最近完整窗口：`77.33% / 1.896`
  - 最新弱窗口：`-10.87% / -0.559`
- `ma50 + up_low_ml55_none25_v220`
  - 全样本超额 Sharpe：`1.392`
  - 最近完整窗口：`64.14% / 1.679`
  - 最新弱窗口：`-9.31% / -0.509`
- 归因上：
  - `ma50 + up_low_ml55_none25_v220` 只在 `17` 个季度里的 `4` 个季度更强
  - 最明显的拖累来自 `trend_up_low_vol`，超额差约 `-100.56%`
- 这说明：
  - 在 `ma50` 框架里继续叠加第一轮那套权重修补，代价明显大于收益；
  - 第四轮权重微调不该再作为立刻要做的下一步。

### 本轮结论
1. 第三轮专项复验已经通过，`ma50 baseline` 继续稳居当前“最强正式修复候选”。
2. `ma50 baseline` 同时优于当前执行主线和次一级备选，而且优势不是单季度异常造成。
3. 在 `ma50` 框架里，继续叠加 `up_low_ml55_none25_v220` 会明显伤害整体表现，因此第四轮权重微调降级为条件触发项。
4. 下一步研究重心应当从“继续调 `trend_up_low_vol` 权重”，切到“复验 `ma50` 的状态边界稳定性与轻量风险控制”。

### 当前决策
1. 执行端默认值继续冻结为：`advanced_ml + liquid500 + next_open`
2. `ma50 baseline` 继续作为当前头号正式修复候选
3. `ma60 + up_low_ml55_none25_v220` 继续保留为次一级备选
4. 下一步直接进入 `ma50` 的状态边界与轻量风险控制复验；第四轮权重微调改为条件触发

## 2026-03-22 执行端第五轮正式复验：`ma50` 边界稳定性与轻量风险控制
### 本轮目标
- 不再继续在 `ma50` 框架里盲调权重，而是先判断：
  - `ma50` 周围的状态边界是否存在更强点
  - 轻量风控是否能在不破坏整体的前提下提供净增益
- 这轮只做两类变量：
  - 状态边界：`ma48 / ma49 / ma50 / ma51 / ma52`
  - 轻量风控：`baseline / stop8 / take20 / stop8_take20`

### 本轮产物
- 边界扫描目录：
  - `daily_research/output/advanced_ml_execution_repair_scan_20260322_formal_round5_ma48_boundary_risk`
  - `daily_research/output/advanced_ml_execution_repair_scan_20260322_formal_round5_ma49_boundary_risk`
  - `daily_research/output/advanced_ml_execution_repair_scan_20260322_formal_round5_ma50_boundary_risk`
  - `daily_research/output/advanced_ml_execution_repair_scan_20260322_formal_round5_ma51_boundary_risk`
  - `daily_research/output/advanced_ml_execution_repair_scan_20260322_formal_round5_ma52_boundary_risk`
- 汇总目录：
  - `daily_research/output/advanced_ml_execution_repair_scan_20260322_formal_round5_boundary_risk_compare`
- 关键归因目录：
  - `daily_research/output/advanced_ml_execution_repair_scan_20260322_formal_round5_boundary_risk_compare/attr_ma50_baseline_vs_ma48_baseline`
  - `daily_research/output/advanced_ml_execution_repair_scan_20260322_formal_round5_boundary_risk_compare/attr_ma50_baseline_vs_ma48_take20`
  - `daily_research/output/advanced_ml_execution_repair_scan_20260322_formal_round5_boundary_risk_compare/attr_ma48_baseline_vs_ma48_take20`
  - `daily_research/output/advanced_ml_execution_repair_scan_20260322_formal_round5_boundary_risk_compare/ma48_take20_revalidation`

### 结果一：边界扫描的主结论
- `ma48 baseline`
  - 全样本超额收益：`729.66%`
  - 全样本超额 Sharpe：`2.202`
  - 最近完整窗口：`112.46% / 2.790`
  - 最新弱窗口：`+3.24% / 0.157`
- `ma48 + take20`
  - 全样本超额收益：`746.20%`
  - 全样本超额 Sharpe：`2.239`
  - 最近完整窗口：`116.70% / 2.930`
  - 最新弱窗口：`+3.24% / 0.157`
- `ma50 baseline`
  - 全样本超额收益：`580.31%`
  - 全样本超额 Sharpe：`1.841`
  - 最近完整窗口：`77.33% / 1.896`
  - 最新弱窗口：`-10.87% / -0.559`
- `ma49 / ma51 / ma52` 的基线都明显更差，尤其最新弱窗口重新大幅转负。
- 这说明：
  - `ma48` 的确跑出了比 `ma50` 更强的边界点；
  - 但这个改善不是一个平滑的“附近都更好”，而是一个比较尖锐的局部甜点。

### 结果二：轻量风控的主结论
- 在 `ma50` 框架里：
  - `take20` 只能把最新弱窗口从 `-10.87% / -0.559` 轻微改善到 `-9.66% / -0.504`
  - 但全样本超额 Sharpe 会从 `1.841` 轻微回落到 `1.836`
  - `stop8` 与 `stop8_take20` 都更差
- 在 `ma48` 框架里：
  - `take20` 把全样本超额 Sharpe 从 `2.202` 小幅抬到 `2.239`
  - 但它对 `ma48 baseline` 的增益很小，更多像附加微调，而不是主驱动
- 这说明：
  - 真正有信息量的是“边界从 `ma50` 收到 `ma48`”，不是“轻量风控本身”
  - `take20` 目前只能算边界候选上的次级增强，而不是单独结论

### 结果三：稳定性与集中度
- `ma48 + take20` 相对 `ma50 baseline`
  - 只在 `17` 个季度里的 `5` 个季度更强
  - 有 `8` 个季度反而更弱
  - 最强季度是 `2025Q3`
  - 单一最强季度占正向季度总优势约 `54.92%`
- `ma48 + take20` 相对 `ma48 baseline`
  - 只在 `17` 个季度里的 `2` 个季度更强
  - 总增益很小
  - 最强季度同样是 `2025Q3`
  - 单一最强季度占正向季度总优势约 `72.05%`
- 这说明：
  - `ma48` 这条线虽然数值很强，但当前优势明显更集中；
  - `take20` 的附加收益本身也不够稳定。

### 归因补充
- `ma48 baseline` / `ma48 + take20` 相对 `ma50 baseline` 的主要新增优势：
  - `trend_up_high_vol` 超额差约 `+34.25%`
  - `trend_down_low_vol` 超额差约 `+4.73%`
- 相对 `ma50 baseline`，它们在 `trend_up_low_vol` 反而没有继续扩大优势。
- 这说明：
  - 第五轮跑出来的新信号，核心不是再次强化原先的 `trend_up_low_vol`
  - 而是边界变化后，对 `trend_up_high_vol` 的映射明显变强

### 本轮结论
1. 第五轮已经确认：`ma50` 附近确实存在一个更强的边界点，当前最亮眼的是 `ma48`。
2. 但 `ma48` 的优势并不平滑，`ma49 / ma51 / ma52` 都明显回落，说明它目前更像局部甜点，而不是已经确认的稳定新主线。
3. `ma48 + take20` 是当前数值最强点，但其相对 `ma48 baseline` 的增益本身高度集中，不足以单独晋级。
4. `ma50` 框架内的轻量风控没有提供足够大的净增益，因此当前不改变“`ma50 baseline` 是头号正式修复候选”的主判断。

### 当前决策
1. 执行端默认值继续冻结为：`advanced_ml + liquid500 + next_open`
2. `ma50 baseline` 继续作为当前头号正式修复候选
3. `ma60 + up_low_ml55_none25_v220` 继续保留为次一级备选
4. 新增一条高收益待复验分支：`ma48 baseline`，`ma48 + take20` 作为其附加轻量风控版本保留
5. 下一步不直接切执行默认值，而是先对 `ma48` 做专门稳定性复验与季度集中度诊断

## 2026-03-22 执行端第六轮正式复验：`ma48` 稳定性复验与季度集中度诊断
### 本轮目标
- 不再只看 `ma48` 在第五轮里的聚合指标，而是专门回答三件事：
  - `ma48` 是否只是一个孤立甜点；
  - `ma48` 相对 `ma50` 和左邻 `ma47` 的优势是否足够平滑稳定；
  - `ma48 + take20` 对 `ma48 baseline` 的附加收益是否能够独立成立。

### 本轮产物
- 新补跑左邻边界目录：
  - `daily_research/output/advanced_ml_execution_repair_scan_20260322_formal_round5_ma47_boundary_risk`
- 第六轮稳定性汇总目录：
  - `daily_research/output/advanced_ml_execution_repair_scan_20260322_formal_round6_ma48_stability`
- 关键归因目录：
  - `daily_research/output/advanced_ml_execution_repair_scan_20260322_formal_round6_ma48_stability/attr_ma47_baseline_vs_ma48_baseline`
- 工具：
  - `daily_research/tools/ma48_stability_report.py`

### 结果一：邻域稳定性
- `ma50 baseline`
  - 全样本超额收益：`580.31%`
  - 全样本超额 Sharpe：`1.841`
  - 最近完整窗口：`77.33% / 1.896`
  - 最新弱窗口：`-10.87% / -0.559`
- `ma47 baseline`
  - 全样本超额收益：`626.68%`
  - 全样本超额 Sharpe：`1.955`
  - 最近完整窗口：`96.93% / 2.380`
  - 最新弱窗口：`+1.79% / 0.106`
- `ma48 baseline`
  - 全样本超额收益：`729.66%`
  - 全样本超额 Sharpe：`2.202`
  - 最近完整窗口：`112.46% / 2.790`
  - 最新弱窗口：`+3.24% / 0.157`
- `ma49 baseline`
  - 全样本超额收益：`499.38%`
  - 全样本超额 Sharpe：`1.774`
  - 最近完整窗口：`71.39% / 1.834`
  - 最新弱窗口：`-30.27% / -1.544`
- 这说明：
  - `ma48` 的强势不再像纯随机孤点，因为左邻 `ma47` 也处在较强水平；
  - 但右邻 `ma49` 明显回落，说明当前更像“左侧 `ma47/48` 强带”，而不是一整段平滑抬升的新稳态。

### 结果二：季度集中度
- `ma48 baseline` 相对 `ma50 baseline`
  - 只在 `17` 个季度里的 `5` 个季度更强
  - 有 `8` 个季度反而更弱
  - 最强季度是 `2025Q3`
  - 单一最强季度占正向季度总优势约 `53.86%`
  - Top3 正向季度占比约 `94.20%`
- `ma48 baseline` 相对 `ma47 baseline`
  - 在 `17` 个季度里的 `6` 个季度更强
  - 有 `4` 个季度更弱
  - 最强季度同样是 `2025Q3`
  - 单一最强季度占正向季度总优势约 `71.02%`
  - Top3 正向季度占比约 `91.32%`
- `ma48 + take20` 相对 `ma48 baseline`
  - 只在 `17` 个季度里的 `2` 个季度更强
  - 总增益约 `3.33%`
  - 最强季度仍是 `2025Q3`
  - 单一最强季度占正向季度总优势约 `72.05%`
- 这说明：
  - `ma48` 相对 `ma50` 的优势仍然高度集中，不能直接视为新的稳定主线；
  - `ma48` 相对 `ma47` 的新增优势同样高度集中，真正需要继续诊断的是 `47 -> 48` 这一步为什么只在少数季度显著拉开；
  - `take20` 在 `ma48` 上只能算很小的附加微调，不足以单独构成升级理由。

### 结果三：归因补充
- `ma48 baseline` 相对 `ma47 baseline`
  - 最强季度是 `2025Q3`
  - 超额差约 `+62.43%`
- 市场状态差异：
  - `trend_up_low_vol`：`+65.68%`
  - `trend_down_low_vol`：`-13.17%`
  - `trend_up_high_vol`：`-8.40%`
- 这说明：
  - `47 -> 48` 的新增收益主要来自 `trend_up_low_vol` 的进一步放大；
  - 但它不是一个全状态普适增强，因此更需要继续拆解集中来源，而不是直接晋级执行端。

### 本轮结论
1. `ma48 baseline` 的高收益并非纯随机孤点，因为左邻 `ma47 baseline` 也处在明显更强的区间。
2. 但 `ma48 baseline` 相对 `ma50 baseline`、相对 `ma47 baseline` 的新增优势都高度集中在少数季度，尤其 `2025Q3`。
3. `ma48 + take20` 相对 `ma48 baseline` 的增益很小且更集中，不足以作为独立晋级理由。
4. 因此当前仍不改变“`ma50 baseline` 是头号正式修复候选”的主判断，但高收益待复验分支应从单点 `ma48` 扩展为左侧 `ma47/48` 边界带。

### 当前决策
1. 执行端默认值继续冻结为：`advanced_ml + liquid500 + next_open`
2. `ma50 baseline` 继续作为当前头号正式修复候选
3. `ma60 + up_low_ml55_none25_v220` 继续保留为次一级备选
4. 高收益待复验分支从单点 `ma48 baseline` 扩展为 `ma47/48` 左侧边界带；其中 `ma48 baseline` 仍是当前数值最强点，`ma48 + take20` 作为其附加轻量风控版本保留
5. 下一步不直接切执行默认值，而是先做 `ma47/48` 左侧边界带的稳定性复验，并拆解 `2025Q3` 的集中来源

## 2026-03-22 执行端第七轮正式复验：`ma47/48` 左侧边界带与 `2025Q3` 集中来源诊断
### 本轮目标
- 不再继续只看 `ma48` 单点，而是专门回答两件事：
  - 左侧 `ma47/48` 边界带是否真的是一段值得继续保留的高收益分支；
  - `2025Q3` 的集中增益到底来自状态切换，还是来自同一状态内的持仓槽位替换。

### 本轮产物
- 汇总目录：
  - `daily_research/output/advanced_ml_execution_repair_scan_20260322_formal_round7_ma4748_band`
- 工具：
  - `daily_research/tools/ma4748_band_report.py`
- 关键明细：
  - `focus_quarter_ma48_vs_ma47_monthly.csv`
  - `focus_quarter_ma48_vs_ma50_monthly.csv`
  - `focus_quarter_ma48_vs_ma47_top_days.csv`
  - `focus_quarter_ma48_vs_ma50_top_days.csv`
  - `focus_quarter_ma48_vs_ma47_overlap.csv`
  - `focus_quarter_ma48_vs_ma50_overlap.csv`

### 结果一：左侧边界带仍然成立
- `ma47 baseline`
  - 全样本超额收益：`626.68%`
  - 全样本超额 Sharpe：`1.955`
  - 最近完整窗口：`96.93% / 2.380`
  - 最新弱窗口：`+1.79% / 0.106`
- `ma48 baseline`
  - 全样本超额收益：`729.66%`
  - 全样本超额 Sharpe：`2.202`
  - 最近完整窗口：`112.46% / 2.790`
  - 最新弱窗口：`+3.24% / 0.157`
- `ma50 baseline`
  - 全样本超额收益：`580.31%`
  - 全样本超额 Sharpe：`1.841`
  - 最近完整窗口：`77.33% / 1.896`
  - 最新弱窗口：`-10.87% / -0.559`
- 这说明：
  - 左侧 `ma47/48` 带不是纯随机噪音，因为 `ma47` 与 `ma48` 都明显强于当前稳定候选 `ma50`
  - 但 `ma48` 仍然只是这条高收益分支里的数值最强点，不等于已经取得执行端晋级资格。

### 结果二：`2025Q3` 的集中来源不是状态切换
- `ma48` 相对 `ma47` 的 `2025Q3` compound 超额边际约 `+62.43%`
- `ma48` 相对 `ma50` 的 `2025Q3` compound 超额边际约 `+55.31%`
- 但 `2025Q3` 的 `66` 个交易日全部都处于：
  - `trend_up_low_vol`
  - `regime_on=True`
- 这说明：
  - `2025Q3` 的集中增益不是靠边界切换后“多开了某些状态”
  - 而是同一 `trend_up_low_vol` 状态内部的选股与换仓差异。

### 结果三：`2025Q3` 优势是逐月放大、但仍受少数关键日驱动
- `ma48` 相对 `ma47` 的月度 compound 超额边际：
  - `2025-07`：`+9.98%`
  - `2025-08`：`+13.60%`
  - `2025-09`：`+16.83%`
- `ma48` 相对 `ma50` 的月度 compound 超额边际：
  - `2025-07`：`+7.65%`
  - `2025-08`：`+8.16%`
  - `2025-09`：`+18.84%`
- 但日度集中度仍不低：
  - `ma48` 相对 `ma47` 的 Top5 正向日占 `2025Q3` 正向日总优势约 `41.12%`
  - `ma48` 相对 `ma50` 的 Top5 正向日占比约 `51.25%`
- 关键日期集中在：
  - `2025-08-22`
  - `2025-08-27`
  - `2025-09-01`
  - `2025-09-03`
  - `2025-09-18`
- 这说明：
  - `2025Q3` 的确不是只靠单一天抬起来；
  - 但优势仍然高度依赖少数关键交易日，而不是完全平滑均匀分布。

### 结果四：优势主要来自少数持仓槽位替换
- `ma48` 相对 `ma47` 的 `2025Q3` 平均持仓重叠：
  - Jaccard 约 `0.697`
  - 约 `72.73%` 的日期至少重合 `4` 个名字
- `ma48` 相对 `ma50` 的 `2025Q3` 平均持仓重叠：
  - Jaccard 约 `0.694`
  - 约 `75.76%` 的日期至少重合 `4` 个名字
- 最大正向贡献日上的典型替换包括：
  - `2025-08-22`：`601328.SH` 替换 `601288.SH`
  - `2025-08-27`：`688660.SH` 替换 `603256.SH`
  - `2025-09-01`：`300486.SZ / 601728.SH / 601288.SH` 替换 `601939.SH / 002142.SZ / 600900.SH`
  - `2025-09-03`：`688108.SH` 替换 `600585.SH`
  - `2025-09-18`：`300204.SZ` 替换 `301357.SZ`
- 这说明：
  - `ma48` 的新增优势主要来自少数持仓槽位的替换；
  - 真正值得继续追的是这些槽位替换是否能跨季度复现，而不是继续盲扫边界。

### 本轮结论
1. 左侧 `ma47/48` 边界带应继续保留为高收益待复验分支，因为它整体确实强于 `ma50 baseline`。
2. `2025Q3` 的集中来源已经明确：不是状态切换，而是 `trend_up_low_vol` 内部的选股与换仓差异。
3. `ma48` 的优势不是纯单日噪音，但仍明显依赖少数关键交易日和少数持仓槽位替换。
4. 因此当前仍不改变“`ma50 baseline` 是头号正式修复候选”的主判断；左侧高收益分支的下一步应转入关键槽位复现诊断，而不是继续大范围扫边界。

### 当前决策
1. 执行端默认值继续冻结为：`advanced_ml + liquid500 + next_open`
2. `ma50 baseline` 继续作为当前头号正式修复候选
3. `ma60 + up_low_ml55_none25_v220` 继续保留为次一级备选
4. 高收益待复验分支继续保留为 `ma47/48` 左侧边界带；其中 `ma48 baseline` 仍是当前数值最强点，`ma48 + take20` 作为其附加轻量风控版本保留
5. 下一步不直接切执行默认值，而是先围绕 `trend_up_low_vol` 做关键槽位复现诊断，检查 `2025Q3` 的增益是否能跨季度重复出现

## 2026-03-22 执行端第八轮正式复验：`trend_up_low_vol` 关键槽位复现诊断
### 本轮目标
- 不再只停留在“`2025Q3` 很强”这个现象判断，而是专门验证：
  - `2025Q3` 的关键槽位替换是否能在其他 `trend_up_low_vol` 季度复现；
  - 如果不能复现，`ma47/48` 左侧带到底应该继续作为晋级候选，还是改成纯研究分支。

### 本轮产物
- 汇总目录：
  - `daily_research/output/advanced_ml_execution_repair_scan_20260322_formal_round8_slot_replay`
- 工具：
  - `daily_research/tools/trend_up_low_vol_slot_replay_report.py`
- 关键明细：
  - `ma48_vs_ma47_slot_replay.csv`
  - `ma48_vs_ma50_slot_replay.csv`
  - `ma48_vs_ma47_focus_signature_stocks.csv`
  - `ma48_vs_ma50_focus_signature_stocks.csv`

### 结果一：`2025Q3` 的 top5 槽位签名已经明确
- `ma48` 相对 `ma47` 的 `2025Q3` top5 槽位签名：
  - `301389.SZ`
  - `301488.SZ`
  - `603716.SH`
  - `300436.SZ`
  - `300486.SZ`
- `ma48` 相对 `ma50` 的 `2025Q3` top5 槽位签名：
  - `301357.SZ`
  - `300436.SZ`
  - `301488.SZ`
  - `300476.SZ`
  - `601728.SH`

### 结果二：按当前 top5 槽位签名口径，跨季度复现是零
- 相对 `ma47`
  - 其他季度里完整复现次数：`0`
  - 放宽到“任意 top5 重叠”的季度数：`0`
  - 正边际季度里出现任意重叠的季度数：`0`
- 相对 `ma50`
  - 其他季度里完整复现次数：`0`
  - 放宽到“任意 top5 重叠”的季度数：`0`
  - 正边际季度里出现任意重叠的季度数：`0`
- 这说明：
  - `2025Q3` 的关键槽位签名在当前口径下并不是一个已经跨季度稳定复放的固定模式；
  - `ma47/48` 左侧带的高收益，更像某个季度里对 `trend_up_low_vol` 的局部命中。

### 结果三：stock 级别的重复出现也很弱
- 相对 `ma47`
  - `301389.SZ / 301488.SZ / 603716.SH / 300436.SZ / 300486.SZ` 这 5 个 Q3 槽位名字，在其他季度里没有一次达到 `0.1%` 以上的平均正权重差复现
- 相对 `ma50`
  - `301357.SZ`：其他季度复现 `0` 次
  - `300436.SZ`：其他季度复现 `0` 次
  - `301488.SZ`：其他季度复现 `1` 次，且落在正边际季度 `2025Q4`
  - `300476.SZ`：其他季度复现 `1` 次，但对应季度 `2025Q2` 不是正边际季度
  - `601728.SH`：其他季度复现 `1` 次，但对应季度 `2024Q2` 不是正边际季度
- 这说明：
  - 即使把诊断下沉到单只股票层面，`2025Q3` 的关键槽位也没有表现出稳定的跨季度复现能力；
  - 目前没有证据支持把这组槽位直接固化成新的执行端配置。

### 本轮结论
1. `2025Q3` 的优势来源已经进一步确认：它不是一个可直接跨季度复放的 top5 槽位签名。
2. `ma47/48` 左侧带当前仍可保留为高收益研究分支，但更像“季度特定槽位命中”，而不是已经成熟的正式修复候选。
3. 因此当前不改变“`ma50 baseline` 是头号正式修复候选”的主判断，也不建议继续直接扫左侧边界。
4. 如果后续还要继续推进这条分支，下一步必须把 `2025Q3` 的槽位替换抽象成更稳定的 `trend_up_low_vol` 信号逻辑，并要求它在非 `2025Q3` 季度也能复放；否则就停止该分支晋级。

### 当前决策
1. 执行端默认值继续冻结为：`advanced_ml + liquid500 + next_open`
2. `ma50 baseline` 继续作为当前头号正式修复候选
3. `ma60 + up_low_ml55_none25_v220` 继续保留为次一级备选
4. `ma47/48` 左侧边界带继续保留为高收益研究分支，但当前按“季度特定槽位命中”看待；其中 `ma48 baseline` 仍是当前数值最强点，`ma48 + take20` 作为附加轻量风控版本保留
5. 下一步不直接切执行默认值，也不继续盲扫边界，而是先把 `2025Q3` 的槽位替换抽象成更稳定的 `trend_up_low_vol` 信号逻辑；若无法抽象并跨季度复放，就停止该分支晋级

## 2026-03-22 执行端第九轮正式复验：`trend_up_low_vol` 信号逻辑抽象诊断
### 本轮目标
- 不再停留在“`2025Q3` 有几只关键槽位股票”的复现口径，而是直接回答：
  - 能否把 `2025Q3` 的槽位替换抽象成一个可重复使用的 `trend_up_low_vol` 信号逻辑；
  - 这个信号逻辑能否同时满足“解释 `ma48` 的槽位差异”与“在其他季度继续对未来超额有效”；
  - 如果做不到，是否应按既定停止规则，终止 `ma47/48` 左侧带的执行端晋级。
### 本轮产物
- 汇总目录：
  - `daily_research/output/advanced_ml_execution_repair_scan_20260322_formal_round9_signal_logic`
- 工具：
  - `daily_research/tools/trend_up_low_vol_signal_logic_report.py`
- 关键明细：
  - `ma48_vs_ma47_topday_factor_diff.csv`
  - `ma48_vs_ma50_topday_factor_diff.csv`
  - `selected_slot_logic_factors.csv`
  - `selected_slot_logic_shared_factors.csv`
  - `quarter_rank_ic.csv`
  - `quarter_slot_edge.csv`
  - `rank_ic_summary.csv`
  - `slot_edge_summary.csv`

### 结果一：宽口径抽象 `slot_logic` 不稳定
- 按 `2025Q3` 顶部换仓日自动提炼出的宽口径 `slot_logic` 因子为：
  - `volatility_contraction`
  - `ma_gap_20_60`
  - `volatility_20`
  - `price_volume_divergence`
  - `mom_20`
  - `long_regime_flag`
- 这条宽口径候选的结果是：
  - 季度 RankIC 均值约 `-0.009`
  - 相对 `v2` 只在 `3` 个非 `2025Q3` 季度更强
  - 这些非焦点季度的正向改善约 `95.71%` 集中在单一季度
- 这说明：
  - 直接把 `Q3` 的替换痕迹做成一个更宽的组合信号，会把一批只在单侧对照中出现的因素也带进来；
  - 这种抽象方式无法形成可晋级的稳定排序逻辑。

### 结果二：更克制的 `slot_logic_shared` 也不够支撑晋级
- 只保留 `ma48_vs_ma47` 与 `ma48_vs_ma50` 都共同支持的因子后，最终剩下的 shared 逻辑只有：
  - `volatility_contraction`
- 这条单因子逻辑的结果是：
  - 季度 RankIC 均值约 `0.044`
  - 相对 `v2` 虽在 `5` 个非 `2025Q3` 季度更强，但正向改善约 `82.42%` 集中在 `2024Q3`
  - 在焦点季度 `2025Q3` 本身，反而落后 `v2` 约 `-0.060`
- 这说明：
  - `Q3` 的槽位替换里确实能抽出一点更窄的“低波收缩”信息；
  - 但它不是 `2025Q3` 那轮优势的稳定核心，更像其他季度里偶发有效的局部排序信号。

### 结果三：信号排序与槽位复放没有同时成立
- `slot_logic_shared` 对 `ma48_vs_ma47` 的季度槽位复放：
  - 在非焦点季度里只出现 `2` 个正向槽位边际
  - 且在 `ma48` 其他正边际季度里没有一次稳定对齐
- `slot_logic_shared` 对 `ma48_vs_ma50` 的季度槽位复放：
  - 在非焦点季度里只出现 `3` 个正向槽位边际
  - 其中只有 `1` 个落在 `ma48` 的其他正边际季度
- `slot_logic` 虽然在部分季度里能给出更高的槽位边际，但它自己的未来超额排序已经不稳定。
- 这说明：
  - 当前抽出来的逻辑要么能解释一点槽位差异、却不能稳定解释未来超额；
  - 要么能在别的季度偶尔有排序价值、却不是 `2025Q3` 那轮优势的稳定抽象。

### 本轮结论
1. `2025Q3` 的槽位替换目前仍无法抽象成一个可跨季度复放的稳定 `trend_up_low_vol` 信号逻辑。
2. `ma47/48` 左侧边界带的高收益事实仍然保留，但它不再满足继续晋级执行端的条件。
3. 因此这条分支应按既定停止规则处理：停止执行端晋级，降级为纯研究旁支，而不是继续扫边界或继续堆局部解释。
4. 研究主线应回到 `ma50 baseline` 内部升级，优先从状态专属 horizon 权重开始，而不是继续追逐 `ma47/48` 的季度特定高点。

### 当前决策
1. 执行端默认值继续冻结为：`advanced_ml + liquid500 + next_open`
2. `ma50 baseline` 继续作为当前头号正式修复候选
3. `ma60 + up_low_ml55_none25_v220` 继续保留为次一级备选
4. `ma47/48` 左侧边界带停止晋级执行端，降级为纯研究旁支；已有结论和产物保留，但不再作为当前执行修复候选
5. 下一步研究重心回到 `ma50 baseline` 内部升级，优先做状态专属 horizon 权重，其后再看状态专属 ensemble 权重与 `ma50` 框架内波动阈值微调

## 2026-03-22 执行端第十至十二轮正式复验：`ma50 baseline` 内部升级
### 本轮目标
- 在确认 `ma47/48` 左侧边界带停止晋级后，把研究主线收回到 `ma50 baseline` 内部；
- 依次回答三件事：
  - 状态专属 horizon 权重，是否能以最小改动修复最新弱窗口；
  - 状态专属 ensemble 权重，是否存在“弱窗口改善且不破坏全样本”的干净升级；
  - `ma50` 框架内的简单波动阈值微调，是否还存在有效敏感度。

### 本轮脚本修正
- 在第十轮正式扫描前，先修正了共享扫描器的一个实现问题：
  - 之前 `scan_execution_repair_candidates.py` 会在候选循环外先合成一遍多周期 `ml_score`，导致状态专属 horizon 候选即使写进配置，也不会真的影响候选结果；
  - 因此补充了 `ml_alpha.py` 的 `combine_per_horizon_ml_scores` 与 `rolling_ml_scores_multi_detail`，并让扫描器在候选循环内按各自 `state_horizon_weights` 重新合成 `candidate_ml_score`；
  - 同时把扫描器扩成支持“候选级全量重算”，以便安全复验 `regime_max_annual_vol` 这种会改状态标签与 ML 特征的参数。

### 本轮产物
- 第十轮 `ma50` 状态专属 horizon 权重：
  - `daily_research/output/advanced_ml_execution_repair_scan_20260322_formal_round10_ma50_state_horizon`
- 第十一轮 `ma50` 状态专属 ensemble 权重：
  - `daily_research/output/advanced_ml_execution_repair_scan_20260322_formal_round11_ma50_state_ensemble`
- 第十二轮 `ma50` 波动阈值微调：
  - `daily_research/output/advanced_ml_execution_repair_scan_20260322_formal_round12_ma50_regime_vol`
- 关键脚本：
  - `daily_research/baseline/scan_execution_repair_candidates.py`
  - `daily_research/baseline/ml_alpha.py`

### 结果一：状态专属 horizon 权重不是当前修复主线
- 只改 `trend_up_high_vol` 的 horizon 配比时：
  - `latest_weak` 窗口始终停在基线同一水平，约 `-10.87% / -0.559`
  - 但全样本超额 Sharpe 会从 `1.841` 回落到约 `1.653 ~ 1.712`
- 一旦同时改到 `trend_up_low_vol` 的 horizon 配比：
  - 最新弱窗口会明显恶化，最差一档约退到 `-18.96% / -0.927`
  - 全样本超额 Sharpe 也进一步退到约 `1.412 ~ 1.608`
- 这说明：
  - `ma50 baseline` 下，状态专属 horizon 权重对当前弱窗口没有提供有效修复；
  - 尤其 `trend_up_low_vol` 的 horizon 结构，当前不宜作为下一步第一优先级。

### 结果二：状态专属 ensemble 权重仍有一点信息量，但还没有干净升级
- 纯 `trend_up_high_vol` 的 ensemble 调整：
  - 对最新弱窗口几乎完全无影响；
  - 但会轻微拖累全样本超额 Sharpe。
- 首个真正动到弱窗口的候选是：
  - `trend_up_low_vol=ml0.60/none0.25/v20.15`
  - `trend_up_high_vol=ml0.80/none0.15/v20.05`
- 这条候选的表现是：
  - 最新弱窗口从基线约 `-10.87% / -0.559` 小幅改善到约 `-10.04% / -0.550`
  - 最近完整窗口提升到约 `90.53% / 2.299`
  - 但全样本超额 Sharpe 从 `1.841` 回落到约 `1.753`
  - 全样本超额最大回撤也从约 `-29.11%` 扩到约 `-32.43%`
- 这说明：
  - 当前三条升级线里，只有状态专属 ensemble 还保留一点继续研究的价值；
  - 但第一轮结果还不足以把它直接晋级成 `ma50 baseline` 的正式替代。

### 结果三：`ma50` 简单波动阈值微调没有有效敏感度
- `regime_max_annual_vol=0.30 / 0.31 / 0.32 / 0.33 / 0.34` 五个点的正式收益指标完全一致：
  - 全样本超额 Sharpe 都约 `1.841`
  - 最新弱窗口都约 `-10.87% / -0.559`
- 进一步比对 `regime_state.csv` 与 `actions.csv` 后确认：
  - 阈值变化只改动了少数 `trend_down_low_vol / trend_down_high_vol` 的标签；
  - `regime_on` 一天都没有变化；
  - 持仓权重与交易动作也完全一致。
- 这说明：
  - 在当前 `ma50 + liquid500 + next_open` 框架里，简单波动阈值微调并没有触发到真正影响执行结果的边界；
  - 这条线当前不应继续放在第一优先级。

### 本轮结论
1. `ma50 baseline` 内部升级的首轮正式复验已经完成，原定三条线里，`horizon` 与简单波动阈值都没有跑出可继续优先推进的信号。
2. 状态专属 ensemble 权重仍保留一定信息量，但当前最佳候选仍是“弱窗口略改善、全样本明显退步”的不干净升级。
3. 因此执行端默认值继续冻结为：`advanced_ml + liquid500 + next_open`，`ma50 baseline` 继续保留为头号正式修复候选。
4. 下一步不再按原顺序继续推 `horizon -> ensemble -> vol threshold`，而是收束为：只对 `ma50` 做第二轮状态专属 ensemble 精扫，并先隔离 `trend_up_low_vol` 的权重结构。

### 当前决策
1. 执行端默认值继续冻结为：`advanced_ml + liquid500 + next_open`
2. `ma50 baseline` 继续作为当前头号正式修复候选
3. `ma60 + up_low_ml55_none25_v220` 继续保留为次一级备选
4. `ma47/48` 左侧边界带继续保留为纯研究旁支，不再参与当前执行修复排序
5. 下一步研究重心继续留在 `ma50 baseline` 内部，但顺序更新为：第二轮状态专属 ensemble 精扫优先，且先隔离 `trend_up_low_vol`；状态专属 horizon 权重与简单波动阈值微调暂不再列为第一优先级

## 2026-03-23 执行端默认值切换：`ma50 baseline`
### 本轮目标
- 既然 `ma50 baseline` 已经完成专项复验，并在稳定性上明确优于当前执行主线，就不再停留在“候选”状态；
- 正式把执行端默认口径从原 `ma60` 切到 `ma50 baseline`，并确保：
  - 训练入口默认值完成切换；
  - 出计划入口默认值完成切换；
  - 当前默认模型产物同步重训到 `ma50`；
  - 文档口径从“候选”更新为“当前默认执行”。

### 本轮动作
1. 执行端入口参数切换：
   - `daily_research/execution/entrypoint_utils.py`
   - `daily_research/execution/update_model.py`
   - `daily_research/execution/run_trade_plan.py`
   - 新增统一默认注入：`--regime-ma-window 50`
2. 修正了执行端脚本直接按路径运行时的入口缺口：
   - `update_model.py` / `run_trade_plan.py` 在最顶部先补 `sys.path`
   - 因此 `python daily_research/execution/update_model.py ...` 与 `python daily_research/execution/run_trade_plan.py ...` 现在都可以直接运行
3. 补强模型元数据：
   - `daily_research/baseline/train_trade_model.py`
   - 默认模型 `json` 现在会写出：
     - `regime_ma_window`
     - `regime_vol_window`
     - `regime_max_annual_vol`
     - `regime_allowed_quadrants`

### 本轮执行结果
1. 已按新的执行默认值重训默认模型产物：
   - `daily_research/execution/models/latest_ml_model.joblib`
   - `daily_research/execution/models/latest_ml_model.json`
2. 当前默认模型元数据已明确写明：
   - `trained_at = 2026-03-23 00:32:27`
   - `regime_ma_window = 50`
   - `regime_vol_window = 20`
   - `regime_max_annual_vol = 0.32`
   - `regime_allowed_quadrants = [trend_up_low_vol, trend_up_high_vol]`
3. 以脚本路径方式完成了执行端冒烟验证：
   - `python daily_research/execution/update_model.py ...`
   - `python daily_research/execution/run_trade_plan.py ... --output-dir daily_research/output/ma50_execution_switch_smoke --experiment-tag ma50_switch_smoke_20260323`
4. 冒烟验证已通过：
   - 训练入口正常写出 `latest_ml_model.joblib/json`
   - 出计划入口正常完成推理并输出到：
     - `daily_research/output/ma50_execution_switch_smoke/ma50_switch_smoke_20260323`

### 本轮结论
1. `ma50 baseline` 已从“头号正式修复候选”正式晋级为当前执行默认口径。
2. 当前执行端默认值已不再是原 `ma60` 口径，而是：`advanced_ml (ma50 baseline) + liquid500 + next_open`。
3. 后续研究主线不再讨论“要不要切到 ma50”，而是直接围绕当前默认执行 `ma50 baseline` 做增量优化。
4. 原 `ma60 + up_low_ml55_none25_v220` 保留为次一级回滚参考，但不再作为当前默认执行主线。

### 当前决策
1. 执行端默认值已切换为：`advanced_ml (ma50 baseline) + liquid500 + next_open`
2. `ma50 baseline` 已正式晋级为当前执行默认口径
3. `ma60 + up_low_ml55_none25_v220` 保留为次一级回滚备选
4. `ma47/48` 左侧边界带继续保留为纯研究旁支，不参与当前执行默认值排序
5. 下一步研究重心继续留在当前执行默认 `ma50 baseline` 内部，优先做第二轮状态专属 ensemble 精扫，并先隔离 `trend_up_low_vol`

## 2026-03-23 执行端模型治理补强：验证指标落盘 + 过期拦截
### 本轮目标
- 解决两个执行端治理空缺：
  - 默认模型产物只有 `train_summary`，没有历史验证摘要，导致“模型是否仍有效”不能直接从产物上看；
  - 出计划时只检查模型文件是否存在，不检查模型是否过期。

### 本轮动作
1. 在 `daily_research/baseline/train_trade_model.py` 中补了默认模型滚动验证摘要：
   - 沿当前默认口径跑历史滚动 ML 验证；
   - 采用滚动 RankIC 摘要作为默认验证指标；
   - 默认 `21` 个交易日一个历史重训块；
   - 结果写入 `latest_ml_model.json -> validation_summary`
2. 在 `daily_research/baseline/generate_daily_trade_plan.py` 中补了模型新鲜度判断：
   - 默认相对当前信号日滞后 `1` 个交易日开始提醒；
   - 默认滞后 `3` 个交易日开始拦截；
   - 可用 `--allow-stale-model` 强制放行
3. 同步把模型验证摘要与模型新鲜度写进：
   - `latest_trade_plan.txt`
   - `plan_summary.json`

### 本轮执行结果
1. 已重新生成默认模型产物：
   - `daily_research/execution/models/latest_ml_model.joblib`
   - `daily_research/execution/models/latest_ml_model.json`
2. 当前默认模型元数据已包含 `validation_summary`：
   - `combined.full` 平均 RankIC 约 `0.107`
   - `combined.recent_126d` 平均 RankIC 约 `0.146`
   - `combined.recent_63d` 平均 RankIC 约 `0.212`
3. 已完成正常执行冒烟：
   - `daily_research/output/model_validation_smoke/model_validation_smoke_20260323`
   - `latest_trade_plan.txt` 已能显示模型最新数据日、模型新鲜度、模型验证摘要
4. 已完成过期拦截冒烟：
   - 构造临时过期模型元数据 `latest_data_date=2026-03-20`
   - 在 `--stale-model-max-trading-days 1` 下，执行端已按预期抛出 `RuntimeError` 并中止出计划

### 本轮结论
1. 默认模型产物现在不再只是“能加载”，而是可以直接看到训练摘要和滚动验证摘要。
2. 执行端现在不再只是“有模型就继续”，而是具备了最基本的模型过期提醒与拦截能力。
3. 当前执行主线仍保持为：`advanced_ml (ma50 baseline) + liquid500 + next_open`，后续优化可以直接建立在这套更可观测的执行底座上。

## 2026-03-23 执行端易用性修正：`current_positions.csv` 升级为可选账号快照
### 本轮目标
- 修正执行端文档与实际使用上的两个摩擦点：
  - 原 README 把“更新持仓”和“准备次日开盘可用现金”拆成了两步，逻辑重复；
  - 日常执行时还要求命令行手填 `--cash`，不利于直接模拟真实账号状态。

### 本轮动作
1. 在 `daily_research/baseline/generate_daily_trade_plan.py` 中补了账号快照读取：
   - `current_positions.csv` 继续兼容原来的老格式：`stock,shares,cost_price`
   - 同时支持新的账号快照格式：同一文件内写 `account` 行现金与 `position` 行持仓
2. `--cash` 现在改成可选覆盖参数：
   - 不传时，优先读取 `current_positions.csv` 中 `account -> available_cash`
   - 传了 `--cash` 时，命令行显式覆盖文件里的现金
3. 在计划摘要与 `latest_trade_plan.txt` 中补了现金来源说明：
   - 会写明本次现金来自命令行覆盖、账号快照，还是“未提供按 0 处理”
4. 更新样例与说明：
   - `daily_research/execution/current_positions.example.csv`
   - `daily_research/execution/README.md`

### 新的推荐账号快照格式
```csv
record_type,stock,shares,cost_price,available_cash
account,,,,200000
position,600000.SH,1000,10.52,
position,600036.SH,800,42.10,
position,000001.SZ,1200,12.38,
```

### 本轮结论
1. 执行端现在不必再把“持仓更新”和“现金输入”拆成两步。
2. 日常更推荐只维护一份 `current_positions.csv`，直接把账号现金和持仓一起写进去。
3. `--cash` 仍然保留，但定位变成“临时覆盖”，而不是每天必须手填的常规入口。

## 2026-03-23 执行端第十三轮正式复验：`ma50` 状态专属 ensemble 第二轮精扫（先隔离 `trend_up_low_vol`）
### 本轮目标
- 既然当前执行默认值已经切到 `ma50 baseline`，这轮不再同时调两个上涨状态，而是先把变量收干净：
  - 固定 `trend_up_high_vol` 回到 baseline；
  - 只围绕 `trend_up_low_vol` 的 `ML / none / v2` 配比，做第二轮细扫；
  - 重点确认：第一轮里“弱窗口略有改善”的信号，在隔离 `trend_up_high_vol` 后是否还能成立，以及能否跑出更干净的正式候选。

### 本轮产物
- 汇总目录：
  - `daily_research/output/advanced_ml_execution_repair_scan_20260323_formal_round13_ma50_state_ensemble_round2_low_only`
- 汇总表：
  - `daily_research/output/advanced_ml_execution_repair_scan_20260323_formal_round13_ma50_state_ensemble_round2_low_only/repair_scan_summary.csv`
- 归因目录：
  - `daily_research/output/advanced_ml_execution_repair_scan_20260323_formal_round13_ma50_state_ensemble_round2_low_only/attr_baseline_vs_up_low_ml62_none23_v215`
  - `daily_research/output/advanced_ml_execution_repair_scan_20260323_formal_round13_ma50_state_ensemble_round2_low_only/attr_baseline_vs_up_low_ml61_none24_v215`
- 关键脚本：
  - `daily_research/baseline/scan_execution_repair_candidates.py`

### 结果一：只隔离 `trend_up_low_vol` 后，弱窗口修复信号仍然存在
- 本轮 baseline（当前执行默认 `ma50 baseline`）：
  - 全样本超额 Sharpe 约 `1.869`
  - 最近完整窗口约 `77.68% / 1.904`
  - 最新弱窗口约 `-10.70% / -0.550`
- 弱窗口修复最强点出现在：
  - `up_low_ml62_none23_v215`
  - 全样本超额 Sharpe 约 `1.775`
  - 最近完整窗口约 `91.04% / 2.279`
  - 最新弱窗口约 `-7.67% / -0.413`
- 相对更平衡的候选是：
  - `up_low_ml61_none24_v215`
  - 全样本超额 Sharpe 约 `1.792`
  - 最近完整窗口约 `91.54% / 2.337`
  - 最新弱窗口约 `-8.05% / -0.451`
- 这说明：
  - 第一轮里“`trend_up_low_vol` 权重仍有修复信息量”的判断没有被推翻；
  - 而且在把 `trend_up_high_vol` 固定回 baseline 后，这条信号反而更清楚。

### 结果二：但当前仍没有跑出“弱窗口改善 + 全样本不伤”的干净升级
- 虽然 `up_low_ml62_none23_v215 / up_low_ml61_none24_v215` 都显著改善了最近完整窗口和最新弱窗口：
  - 但两者的全样本超额 Sharpe 仍分别从 baseline 的约 `1.869` 回落到约 `1.775 / 1.792`
  - 全样本超额最大回撤也分别扩到约 `-31.28% / -32.62%`，都差于 baseline 的约 `-29.11%`
- 更激进地把 `ML` 压到 `0.59` 以下后，结果开始明显恶化：
  - `up_low_ml59_none25_v216` 的最新弱窗口已经退到约 `-14.67% / -0.792`
  - `up_low_ml58_none25_v217` 更差，约 `-19.46% / -1.018`
- 这说明：
  - `trend_up_low_vol` 的第二轮细扫已经把有效区间压缩到了 `0.60 ~ 0.62` 附近；
  - 再继续往下压 `ML` 主导权，并不会持续改善，反而会把这条线推回失效区。

### 结果三：当前最强候选的增益仍然带有季度集中迹象
- `up_low_ml62_none23_v215` 相对 baseline：
  - 只在 `17` 个季度里的 `7` 个季度更强
  - 最强季度是 `2026Q1`，超额差约 `+9.04%`
  - 最弱季度是 `2024Q4`，超额差约 `-16.58%`
- `up_low_ml61_none24_v215` 相对 baseline：
  - 也只在 `17` 个季度里的 `7` 个季度更强
  - 最强季度同样是 `2026Q1`，超额差约 `+11.19%`
  - 最弱季度同样落在 `2024Q4`，超额差约 `-16.47%`
- 状态归因也表明：
  - 这轮差异几乎全部来自 `trend_up_low_vol`
  - `trend_up_high_vol` 与两个下跌象限基本没变
- 这说明：
  - “先隔离 `trend_up_low_vol`”这一步是对的；
  - 但当前最佳候选仍然可能带有较强的季度集中性，不能直接按正式升级处理。

### 本轮结论
1. `ma50 baseline` 内部第二轮状态专属 ensemble 精扫已经完成，且确认：真正还有继续研究价值的，确实是 `trend_up_low_vol` 这条线。
2. 目前保留下来的两个候选是：
   - `up_low_ml62_none23_v215`：偏弱窗口修复最强；
   - `up_low_ml61_none24_v215`：偏近期窗口相对更平衡。
3. 但它们都还不是“弱窗口改善且全样本不伤”的干净升级，因此当前执行默认值不切换，继续保持 `advanced_ml (ma50 baseline) + liquid500 + next_open`。
4. 下一步不再继续盲扫更大的 `trend_up_low_vol` 权重网格，而是先对这两档候选做季度集中度与 `2026Q1` 归因诊断；如果确认增益仍主要集中在单一季度，就停止这条 ensemble 权重线晋级。

### 当前决策
1. 执行端默认值继续保持为：`advanced_ml (ma50 baseline) + liquid500 + next_open`
2. `ma50 baseline` 继续作为当前执行默认口径
3. `ma60 + up_low_ml55_none25_v220` 继续保留为次一级回滚备选
4. `up_low_ml62_none23_v215` 与 `up_low_ml61_none24_v215` 作为本轮保留的两档研究候选，但暂不晋级执行端
5. 下一步先做这两档候选的季度集中度与 `2026Q1` 归因诊断；若确认仍属季度集中驱动，则停止这条 ensemble 权重线继续晋级


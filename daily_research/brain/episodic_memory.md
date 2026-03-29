# 研究日志

## 总览
- 2026-03-14：初始化 `daily_research` baseline。
- 2026-03-17：完成第一阶段研究框架扩展。
- 当前项目已经形成三层分工：
  - `baseline / advanced_ml`
  - `execution`
  - `deep_alpha`
- 当前执行端冻结为：`advanced_ml (ma50 baseline, lgbm) + liquid500 + next_open`
- `deep_alpha` 仍是正式研究主线之一，但当前最近待决策事项已经切到执行端升级 shortlist 的最终判决。

## 阅读说明
- 本文件只保留按时间顺序排列的实验记录、结果与结论。
- 当前状态与使用入口见 `semantic_memory.md`。
- 项目背景、当前瓶颈与未来方向见 `project_map.md`。
- 当前默认决策、升级 shortlist、优先级与停止规则见 `working_memory.md`。
- 当前运行基线与解释器口径见 `environment_model.md`。
- 执行流程与日常操作细节见 `action_system.md`。

## 阶段索引
- `2026-03-17 ~ 2026-03-18`
  - 基线收敛：`score + 5d`、市场状态过滤、`up_low_breakout_v2`
- `2026-03-18 ~ 2026-03-20`
  - `advanced_ml` 主线形成，并接入执行端
- `2026-03-19 ~ 2026-03-22`
  - `deep_alpha` 从烟测走向正式研究框架
- `2026-03-22 ~ 2026-03-24`
  - 项目治理：入口去重、Git 纳管、归档规则、文档分工、执行口径固化
- `2026-03-25`
  - 坏市场专项目标澄清与首批防守候选收口
- `2026-03-26 ~ 2026-03-28`
  - `advanced_ml (ma50 baseline, lgbm)` 状态专属候选、正式参数矩阵与长窗口 pair revalidation 收敛

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

## 2026-03-23 执行端第十四轮正式诊断：`ma50` ensemble 候选季度集中度与 `2026Q1` 归因
### 本轮目标
- 对 round13 保留下来的两档候选做最后一步正式诊断：
  - `up_low_ml62_none23_v215`
  - `up_low_ml61_none24_v215`
- 直接回答两个问题：
  - 它们相对 `ma50 baseline` 的改进是否仍主要集中在少数季度；
  - 若焦点季度为 `2026Q1`，增益究竟来自稳定的季度级增强，还是少数日期与少数槽位放大。

### 本轮产物
- 汇总目录：
  - `daily_research/output/advanced_ml_execution_repair_scan_20260323_formal_round14_ma50_ensemble_q1_diagnosis`
- 关键文件：
  - `summary_rows.csv`
  - `repair_vs_baseline_quarterly_compare.csv`
  - `balance_vs_baseline_quarterly_compare.csv`
  - `repair_focus_quarter_monthly.csv`
  - `balance_focus_quarter_monthly.csv`
  - `repair_focus_quarter_top_days.csv`
  - `balance_focus_quarter_top_days.csv`
  - `report.json`
  - `report.md`
- 关键工具：
  - `daily_research/tools/ma50_ensemble_q1_report.py`

### 结果一：两条候选都确认存在较强季度集中度
- `up_low_ml62_none23_v215` 相对 baseline：
  - 只在 `17` 个季度里的 `7` 个季度更强
  - 最佳季度 `2026Q1` 占正向季度总优势约 `45.27%`
  - Top3 季度占比约 `81.16%`
  - 季度正向优势 HHI 约 `0.285`
- `up_low_ml61_none24_v215` 相对 baseline：
  - 也只在 `17` 个季度里的 `7` 个季度更强
  - 最佳季度 `2026Q1` 占正向季度总优势约 `52.20%`
  - Top3 季度占比约 `86.74%`
  - 季度正向优势 HHI 约 `0.341`
- 这说明：
  - 两条候选都不是“多季度平滑抬升”的修复；
  - 其中 `up_low_ml61_none24_v215` 比 `up_low_ml62_none23_v215` 还要更集中。

### 结果二：`2026Q1` 的增量主要堆在 `2026-01`
- `up_low_ml62_none23_v215`：
  - `2026-01` compound 超额边际约 `+8.93%`
  - `2026-02` 反而回吐约 `-4.75%`
  - `2026-03` 仅修复约 `+3.41%`
- `up_low_ml61_none24_v215`：
  - `2026-01` compound 超额边际约 `+12.99%`
  - `2026-02` 回吐约 `-6.79%`
  - `2026-03` 仅修复约 `+3.41%`
- 这说明：
  - 即使把最佳季度拆到月度，增量也不是均匀分布；
  - 真正的放大主要集中在 `2026-01`，而不是整个 `2026Q1` 都持续占优。

### 结果三：焦点季度内的增益仍主要来自少数日期与少数槽位替换
- `up_low_ml62_none23_v215` 在 `2026Q1`：
  - Top5 正向日占正向日总优势约 `79.11%`
  - Top10 正向日占比约 `97.26%`
  - 平均持仓重叠 Jaccard 约 `0.714`
  - 约 `54%` 的日期仍至少与 baseline 重合 `4` 个名字
- `up_low_ml61_none24_v215` 在 `2026Q1`：
  - Top5 正向日占比约 `79.12%`
  - Top10 正向日占比约 `96.02%`
  - 平均持仓重叠 Jaccard 约 `0.697`
  - 约 `52%` 的日期仍至少与 baseline 重合 `4` 个名字
- 焦点季度里的正向差异主要发生在 `trend_up_low_vol`，但并不是整季整套组合重写：
  - 更多还是少数日期放大；
  - 再叠加少数槽位替换完成。

### 本轮结论
1. `up_low_ml62_none23_v215` 与 `up_low_ml61_none24_v215` 的正式季度集中度与 `2026Q1` 归因诊断已经完成。
2. 结论可以正式落地为：这条 `trend_up_low_vol` ensemble 权重线仍然属于季度集中驱动，不满足继续晋级执行端的条件。
3. 因此这条线到此停止晋级执行端；已有扫描结果、归因结果和焦点季度诊断全部保留，但仅作为研究附录，不再继续扩展权重网格。
4. 当前执行默认值继续保持为：`advanced_ml (ma50 baseline) + liquid500 + next_open`。
5. 下一步若继续做执行端 ML 增量优化，优先切到 `ma50` 口径下的模型族对照或状态专属模型研究，而不是继续扫这条 ensemble 权重线。

### 当前决策
1. 执行端默认值继续保持为：`advanced_ml (ma50 baseline) + liquid500 + next_open`
2. `ma50 baseline` 继续作为当前执行默认口径
3. `ma60 + up_low_ml55_none25_v220` 继续保留为次一级回滚备选
4. `up_low_ml62_none23_v215` 与 `up_low_ml61_none24_v215` 的诊断已完成，并确认仍属季度集中驱动；两者保留为研究附录，但停止继续晋级执行端
5. 下一步若继续做执行端 ML 增量优化，优先切到 `ma50` 口径下的模型族对照或状态专属模型研究

## 2026-03-24 执行端第十五轮正式复验：`ma50` 口径模型族对照（`histgb / lgbm / etr`）
> 注：这段高收益结论属于 `2026-03-24` 的旧执行口径，不等于当前 live 执行端的收益预期。
> 当时脚本默认起点仍是 `2022-01-01`，且使用的是旧 `market_features(7)` 与全局 `ML / none / v2 = 0.70 / 0.20 / 0.10` 集成；
> `2026-03-28` 底层状态和 `market_features` 已扩到 `24` 个，当前 live 默认也已切到 `trend_up_low_vol` 状态专属 `v250`，因此这里的高收益只能作为“旧模型族对照结论”，不能直接外推到今天的执行端。

### 本轮目标
- 既然 `trend_up_low_vol` 的 ensemble 权重线已经确认停止晋级，就把执行端 ML 增量优化的主线切到当前真实执行口径下的模型族对照：
  - 固定 `ma50 + rolling liquid500 + next_open`
  - 不再沿用旧的 `ma60` 模型族结论
  - 直接回答：当前默认 `histgb` 是否仍然是最合适的执行模型族
- 本轮也把之前超时中断的 `etr` 单独补跑完成，避免三家模型里只留两家半结论。

### 本轮产物
- 汇总目录：
  - `daily_research/output/advanced_ml_model_family_compare_20260323_formal_ma50_execution`
- 关键文件：
  - `model_family_compare_summary.csv`
  - `model_family_compare_report.json`
  - `model_family_compare_report.md`
  - `histgb_vs_lgbm/overall_comparison.csv`
  - `histgb_vs_lgbm/quarterly_comparison.csv`
  - `histgb_vs_lgbm/quadrant_comparison.csv`
- 关键脚本：
  - `daily_research/baseline/compare_ml_model_families.py`
  - `daily_research/baseline/analyze_advanced_ml_comparison.py`

### 结果一：`lgbm` 在当前执行口径下显著强于当前默认 `histgb`
> 2026-03-28 追加澄清：
> 这里的 `887.75%` 是旧口径下的“全样本超额总收益”，不是年化收益，也不是“一年翻数倍”。
> 同段里的 `127.99% / 2.816` 也应读作“最近完整窗口超额总收益 / 超额 Sharpe”，不是“窗口年化 / Sharpe”。
> 另外，这轮模型族对照脚本当时默认仍是 `start_date=20220101`、全局 blend `0.70 / 0.20 / 0.10`，且发生在 `market_features(7)` 旧系统里；
> 2026-03-28 切到 `market_features(24)` 后，当前 live 口径必须以 `advanced_ml_attack_defense_controller_20260328_formal_r3_weightgrid_focus`
> 与 `market_feature_profile_compare_20260328_formal_r1` 为准，不能把这里的旧高收益直接外推成今天执行端的收益预期。
- `lgbm`：
  - 全样本超额收益约 `887.75%`
  - 全样本超额 Sharpe 约 `2.300`
  - 最近完整窗口 `2025-03-07 -> 2026-03-19` 约 `127.99% / 2.816`
  - 最新弱窗口 `2025-09-05 -> 2026-03-19` 约 `9.79% / 0.475`
- 当前默认 `histgb`：
  - 全样本超额收益约 `388.95%`
  - 全样本超额 Sharpe 约 `1.487`
  - 最近完整窗口约 `61.33% / 1.402`
  - 最新弱窗口约 `-8.46% / -0.423`
- 这说明：
  - 在当前已经切换到 `ma50 baseline` 的执行口径下，`histgb` 不再是最强模型族；
  - `lgbm` 不只是弱窗口更好，而是全样本、最近完整窗口、最新弱窗口三层都明显更强。

### 结果二：`etr` 补跑完成，但仍不是头号 challenger
- `etr` 正式补跑结果为：
  - 全样本超额收益约 `274.63%`
  - 全样本超额 Sharpe 约 `1.443`
  - 最近完整窗口约 `41.74% / 1.491`
  - 最新弱窗口约 `6.46% / 0.433`
- 这说明：
  - `etr` 的确比当前默认 `histgb` 更能修复最新弱窗口；
  - 但它在全样本与最近完整窗口上都明显落后于 `lgbm`，且全样本超额 Sharpe 也低于 `histgb`；
  - 因此 `etr` 只保留为正式研究附录，不再作为头号模型族升级候选。

### 结果三：`lgbm` 的领先不是单季度孤点
- `lgbm` 相对 `histgb` 的季度对照结果：
  - 在 `17` 个季度里有 `9` 个季度更强
  - `8` 个季度持平
  - `0` 个季度更弱
- 季度集中度摘要：
  - 最强季度为 `2025Q3`
  - 该季度占正向季度总优势约 `29.67%`
  - Top3 季度占比约 `74.12%`
  - 正向季度 HHI 约 `0.214`
- 状态归因结果：
  - 主要新增优势来自 `trend_up_low_vol`
  - `trend_up_high_vol` 也有正向增益
  - 并不存在“一个上涨状态修好、另一个上涨状态反而更差”的问题
- 这说明：
  - 这次 `lgbm` 的领先不是“少数季度抬起来、其他季度更差”；
  - 更像是从 `2024Q1` 起，在当前 `ma50` 执行框架里持续把 `histgb` 拉开。

### 本轮结论
1. `ma50` 口径下的正式模型族对照已经完成，当前最强模型族已从默认 `histgb` 明确切换为 `lgbm`。
2. `etr` 正式补跑完成后，结论更新为：它能修复弱窗口，但整体不如 `lgbm`，因此只保留为研究附录。
3. `lgbm` 相对 `histgb` 的领先已经具备跨窗口、跨季度的一致性，不再只是旧 `ma60` 口径下那个“值得继续观察”的 challenger。
4. 但当前执行默认模型暂不直接切换，先进入 `ma50 + lgbm` 的切换前复核与执行端烟测，再决定是否正式替换默认模型。

### 当前决策
1. 执行端默认值继续保持为：`advanced_ml (ma50 baseline) + liquid500 + next_open`
2. 当前默认模型族继续保持为：`histgb`
3. `lgbm` 已正式晋级为当前 `ma50` 执行口径下的头号模型族升级候选
4. `etr` 正式补跑完成，但只保留为研究附录，不再作为主 challenger
5. 下一步若继续做执行端 ML 增量优化，优先顺序更新为：
   - 先对 `ma50 + lgbm` 做切换前复核与执行端烟测
   - 再决定是否还有必要进入状态专属模型研究

## 2026-03-24 执行端第十六轮正式复核：`ma50 + lgbm` 切换前复核与第一轮执行端烟测
### 本轮目标
- 既然 `lgbm` 已在当前 `ma50` 执行口径下跑成头号模型族升级候选，这轮不直接切默认值，而是先做切换前复核：
  - 生成独立的 `lgbm` 候选模型产物，不覆盖默认 `latest_ml_model.*`
  - 确认候选产物也能完整走通执行端 `update_model.py -> run_trade_plan.py`
  - 检查执行端是否能正常读取候选 meta 里的 `validation_summary` 与模型新鲜度
  - 再判断：当前是否已经足够进入默认模型切换

### 本轮产物
- 候选模型目录：
  - `daily_research/output/ma50_lgbm_switch_review/models`
- 候选模型产物：
  - `ma50_lgbm_candidate.joblib`
  - `ma50_lgbm_candidate.json`
- 烟测目录：
  - `daily_research/output/ma50_lgbm_switch_smoke/ma50_lgbm_switch_smoke_20260324`
  - `daily_research/output/ma50_lgbm_switch_smoke/latest_trade_plan.txt`
- 复核摘要：
  - `daily_research/output/ma50_lgbm_switch_review/switch_review_summary.json`
  - `daily_research/output/ma50_lgbm_switch_review/switch_review_summary.md`

### 结果一：独立 `lgbm` 候选产物已成功生成，且验证摘要继续强于默认 `histgb`
- 候选产物训练完成时间：
  - `trained_at = 2026-03-24 11:27:56`
- 候选产物口径：
  - `ma50 + liquid500 + next_open`
  - `ml_model_family = lgbm`
- 候选 `lgbm` 的滚动验证摘要：
  - `full IC 0.121`
  - `recent126d IC 0.152`
  - `recent63d IC 0.219`
- 当前默认 `histgb` 的对应摘要：
  - `full IC 0.107`
  - `recent126d IC 0.146`
  - `recent63d IC 0.212`
- 这说明：
  - `lgbm` 不只是在回测收益上更强，离线滚动验证摘要也继续优于当前默认 `histgb`
  - 当前 `lgbm` 候选产物已经具备进入执行端复核的基础质量

### 结果二：第一轮执行端烟测已跑通，但当前只覆盖到 `regime_off` 卖出路径
- 烟测命令已用独立候选产物跑通：
  - `update_model.py` 显式输出到独立 `artifact-path / artifact-meta-path`
  - `run_trade_plan.py` 显式读取该候选 `model-artifact`
- 烟测摘要确认：
  - 模型新鲜度 `fresh`
  - `trading_day_lag = 0`
  - 执行端已正常读取候选 meta 中的 `validation_summary`
- 当前烟测信号日为：
  - `2026-03-23`
  - 市场状态 `trend_down_low_vol`
  - `regime_off`
- 因此本次烟测只走到了“禁止开新仓时的卖出路径”，计划结果为：
  - 卖出 `002843.SZ` `800` 股

### 结果三：候选 `lgbm` 与当前默认计划在本次烟测日没有引入额外执行漂移
- 当前默认计划与候选 `lgbm` 计划在 `2026-03-23` 的动作完全一致：
  - 都只给出一笔卖出 `002843.SZ` `800` 股
- 差异主要体现在分数层：
  - 默认 `histgb` 对该标的的 `final_score` 约 `2.447`
  - 候选 `lgbm` 对该标的的 `final_score` 约 `2.767`
- 这说明：
  - 在当前这个 `regime_off` 信号日上，切到 `lgbm` 不会导致额外的执行动作漂移
  - 但这仍不足以证明买入路径也完全稳定，因为今天没有覆盖到 `regime_on` 下的新开仓场景

### 本轮结论
1. `ma50 + lgbm` 的切换前复核已经完成第一步：独立候选产物与执行端烟测都已跑通。
2. 当前候选 `lgbm` 不仅正式回测强于默认 `histgb`，滚动验证摘要也继续优于默认模型。
3. 但本次烟测落在 `trend_down_low_vol`，只覆盖了 `regime_off` 卖出路径，还不能直接作为最终切换依据。
4. 因此当前执行默认模型暂不切换；下一步应先补一个 `regime_on` 日期的点时烟测，把买入路径也完整验证掉。

### 当前决策
1. 执行端默认值继续保持为：`advanced_ml (ma50 baseline) + liquid500 + next_open`
2. 当前默认模型族继续保持为：`histgb`
3. `lgbm` 继续作为当前头号模型族升级候选
4. 下一步不直接进入状态专属模型研究，而是先补一个 `regime_on` 日期的点时烟测
5. 只有在该点时烟测也通过后，才决定是否正式把默认模型从 `histgb` 切换到 `lgbm`

## 2026-03-24 执行端第十七轮正式复核：`regime_on` 点时烟测补完与买入路径修复
### 本轮目标
- 承接第十六轮尚未完成的切换前复核，直接回答两个问题：
  - `ma50 + lgbm` 在 `regime_on` 场景下是否也能稳定走通真实买入路径；
  - 若点时烟测仍异常，问题究竟来自模型本身，还是执行端计划生成逻辑。

### 本轮产物
- 点时账户快照：
  - `daily_research/output/ma50_lgbm_switch_review/regime_on_account_snapshot.csv`
- 点时模型产物：
  - `daily_research/output/ma50_lgbm_switch_review/models/ma50_histgb_pointtime_20260311.joblib`
  - `daily_research/output/ma50_lgbm_switch_review/models/ma50_histgb_pointtime_20260311.json`
  - `daily_research/output/ma50_lgbm_switch_review/models/ma50_lgbm_pointtime_20260311.joblib`
  - `daily_research/output/ma50_lgbm_switch_review/models/ma50_lgbm_pointtime_20260311.json`
- 修复后点时烟测目录：
  - `daily_research/output/ma50_lgbm_switch_review/histgb_regime_on_smoke_fixed/20260311`
  - `daily_research/output/ma50_lgbm_switch_review/lgbm_regime_on_smoke_fixed/20260311`

### 结果一：前一次 `regime_on` 失败暴露的是执行端买入 bug，不是模型失效
- 原始异常现象是：
  - `plan_summary.json` 显示 `target_position_count = 5`
  - `watchlist.csv` 里已有多只接近 `25%` 的目标仓位
  - 但 `actions_today.csv` 为空，计划文本写成“今日无明确调仓动作”
- 复核后定位到真实原因：
  - `daily_research/baseline/generate_daily_trade_plan.py` 的买入腿循环里，误把 `target_weight_row` 当成了 `target_value`
  - 结果就是买入判断实际在拿 `0.25` 这类权重去和一手股票金额比较，正常候选会被直接跳过
- 这说明：
  - 前一次 `regime_on` 烟测未通过，不能归因到 `histgb` 或 `lgbm`
  - 它首先是一个执行端计划生成 bug

### 结果二：修复后，`histgb / lgbm` 都能在同一 `regime_on` 日期正常生成买单
- 本轮点时信号日固定为：`2026-03-11`
- 当日市场状态为：
  - `trend_up_low_vol`
  - `regime_on = True`
- 修复后 `histgb` 点时计划：
  - 买入 `002470.SZ` `16600` 股
  - 买入 `688800.SH` `500` 股
  - 买入 `300617.SZ` `700` 股
  - 买入 `002843.SZ` `1800` 股
  - 计划后剩余现金约 `9098`
- 修复后 `lgbm` 点时计划：
  - 买入 `002470.SZ` `16600` 股
  - 买入 `000510.SZ` `2600` 股
  - 买入 `688800.SH` `500` 股
  - 买入 `300739.SZ` `1700` 股
  - 计划后剩余现金约 `5584`
- 两边点时产物都保持：
  - `model_freshness = fresh`
  - `trading_day_lag = 0`

### 结果三：切换前执行链路现在已经补全
- 第十六轮已经覆盖了 `2026-03-23` 的 `regime_off` 卖出路径：
  - 默认 `histgb` 与候选 `lgbm` 给出同一笔卖出 `002843.SZ` `800` 股
- 第十七轮又补完了 `2026-03-11` 的 `regime_on` 买入路径：
  - `histgb / lgbm` 都能正常生成多笔买单
  - 且 `lgbm` 的买入组合与 `histgb` 有清晰但可解释的差异，不是执行链路漂移失控
- 这说明：
  - 当前 `ma50 + lgbm` 不只是离线回测与验证摘要更强
  - 在执行端里也已经同时通过了卖出场景和买入场景的烟测复核

### 本轮结论
1. `regime_on` 点时烟测已经补完，且在修复执行端买入 bug 后正式通过。
2. 前一次“有目标仓位却无买单”的异常，已经确认是执行端买入腿 sizing 逻辑问题，不构成对 `lgbm` 的负面证据。
3. 因此当前默认模型从研究侧和执行侧都已满足正式切换到 `lgbm` 的条件。
4. 下一步不再优先进入状态专属模型研究，而是应先完成默认模型从 `histgb` 到 `lgbm` 的正式替换。

### 当前决策
1. 执行端默认值继续保持为：`advanced_ml (ma50 baseline) + liquid500 + next_open`
2. 当前默认模型族仍暂时是：`histgb`
3. 但 `lgbm` 的切换前复核已经完整通过，现已具备正式替换默认模型的条件
4. 下一步优先做默认模型切换：`histgb -> lgbm`
5. 默认模型切换完成后，再决定是否还有必要进入状态专属模型研究

## 2026-03-24 执行端第十八轮正式变更：默认模型 `histgb -> lgbm`
### 本轮目标
- 承接第十七轮已经完成的切换前复核，不再停留在“具备切换条件”，而是把默认执行模型真正从 `histgb` 切换到 `lgbm`：
  - 修改执行端默认入口；
  - 重训默认 `latest_ml_model.*`；
  - 重新跑默认 `run_trade_plan.py`，确认切换后真实默认链路正常。

### 本轮产物
- 默认模型产物：
  - `daily_research/execution/models/latest_ml_model.joblib`
  - `daily_research/execution/models/latest_ml_model.json`
- 默认执行输出：
  - `daily_research/execution/output/20260324`
  - `daily_research/execution/output/latest_trade_plan.txt`

### 结果一：执行端默认入口已切到 `lgbm`
- `daily_research/execution/update_model.py` 现已默认注入：
  - `--ml-model-family lgbm`
- 这意味着：
  - 后续按执行端标准流程运行 `update_model.py` 时，不再需要手动显式补 `--ml-model-family lgbm`
  - 当前默认执行模型族已从“文档建议切换”升级为“入口默认已切换”

### 结果二：默认 `latest_ml_model.*` 已重训为 `lgbm`
- 默认模型元数据当前为：
  - `trained_at = 2026-03-24 16:33:31`
  - `latest_data_date = 2026-03-24`
  - `regime_ma_window = 50`
  - `model_family = lgbm`
- 当前默认滚动验证摘要为：
  - `full IC 0.120`
  - `recent126d IC 0.153`
  - `recent63d IC 0.227`
- 这说明：
  - 当前默认执行产物已经不再是旧的 `histgb`
  - 默认元数据中的验证摘要和新鲜度也都同步更新到了新模型上

### 结果三：切换后的默认执行链路烟测正常
- 用当前真实执行快照重新运行默认 `run_trade_plan.py` 后：
  - 信号日：`2026-03-24`
  - 市场状态：`trend_down_low_vol`
  - 默认计划动作：卖出 `002843.SZ` `800` 股
- 当前计划文件已写回：
  - `daily_research/execution/output/latest_trade_plan.txt`
- 这说明：
  - 默认 `lgbm` 切换后，执行端入口、默认模型产物、默认计划输出三者口径已经重新对齐
  - 本次切换没有引入新的执行异常

### 本轮结论
1. 默认模型 `histgb -> lgbm` 已于 `2026-03-24` 正式完成，不再只是候选结论。
2. 当前执行默认口径正式更新为：`advanced_ml (ma50 baseline, lgbm) + liquid500 + next_open`。
3. 切换后的默认模型产物、模型新鲜度、验证摘要与默认计划输出均已完成同步刷新。
4. 下一步不直接进入状态专属模型研究，而是先判断：在默认 `lgbm` 已经切换完成后，是否还存在值得继续投入的额外 ML 增量空间。

### 当前决策
1. 执行端默认值继续保持为：`advanced_ml + liquid500 + next_open`
2. 其中默认状态边界为：`ma50 baseline`
3. 其中默认模型族现已正式切换为：`lgbm`
4. `ma60 + up_low_ml55_none25_v220` 继续只保留为旧执行口径回滚备选
5. 下一步再决定是否还有必要进入状态专属模型研究

## 2026-03-24 执行端第十九轮正式决策：暂不进入状态专属模型研究
### 本轮目标
- 承接第十八轮默认模型切换后的待决事项，不再继续自动展开新一轮高成本研究，而是先回答：
  - 在默认 `lgbm` 已经切换完成后，当前是否仍有必要立刻进入状态专属模型研究；
  - 还是应该先冻结当前默认口径，观察切换后的真实执行与滚动验证表现。

### 本轮依据
- 当前默认执行口径已经切到：
  - `advanced_ml (ma50 baseline, lgbm) + liquid500 + next_open`
- 当前正式模型族对照结果显示：
  - `lgbm` 全样本超额 Sharpe 约 `2.300`
  - `histgb` 全样本超额 Sharpe 约 `1.487`
  - `lgbm` 相对 `histgb` 在 `17` 个季度里有 `9` 个季度更强、`8` 个季度持平、`0` 个季度更弱
- 当前正式按状态对照结果显示：
  - `trend_up_low_vol` 中，`lgbm` 相对 `histgb` 的超额边际约 `+164.71%`
  - `trend_up_high_vol` 中，`lgbm` 相对 `histgb` 的超额边际约 `+15.78%`
- 历史 `ma50` 内部升级结论显示：
  - 状态专属 horizon 权重线没有形成稳定增量
  - 状态专属 ensemble 权重线最终确认仍属季度集中驱动，已停止晋级执行端

### 本轮判断
1. 当前没有足够证据支持“默认 `lgbm` 刚切完就立刻进入状态专属模型研究”。
2. 原因不是状态专属模型永远没价值，而是当前最重要的两个真实开仓状态 `trend_up_low_vol / trend_up_high_vol`，已经在正式模型族对照里被 `lgbm` 同步改善。
3. 与此同时，现有状态专属 horizon / ensemble 研究并没有拿出比当前默认 `lgbm` 更干净、更稳的升级证据。
4. 因此这一步的正式决策应落为：暂不进入状态专属模型研究，先冻结当前默认执行口径，观察切换后的真实执行与滚动验证表现。

### 当前决策
1. 执行端默认值继续保持为：`advanced_ml + liquid500 + next_open`
2. 其中默认状态边界保持为：`ma50 baseline`
3. 其中默认模型族保持为：`lgbm`
4. 当前不进入状态专属模型研究
5. 只有在后续重新出现明确的状态内失衡证据时，才重启状态专属模型研究
## 2026-03-24 执行端第二十轮正式诊断：`v2` 规则层首轮减法诊断
### 本轮目标
- 承接“`none / v2` 规则层值得做的 3 个小升级点”，先做第一步：
  - 不急着加新因子；
  - 先回答 `v2` 里哪些因子/分组是真贡献，哪些可能已经变成过重负担。
- 诊断口径固定为当前执行默认框架的规则层隔离评估：
  - `ma50`
  - `rolling liquid500`
  - `next_open`
  - `holding_count=5`
  - `max_style_weight=0.50`

### 本轮产物
- 汇总目录：
  - `daily_research/output/v2_rule_ablation_20260324_formal_round1_fixed`
- 关键工具：
  - `daily_research/tools/v2_rule_ablation_report.py`
- 关键汇总：
  - `summary_rows.csv`
  - `quarter_rankic_summary.csv`
  - `report.json`
  - `report.md`

### 过程补记：顺手修复一个 `next_open` 基准对齐 bug
- 在正式开跑这轮 ablation 时，暴露出 `daily_research/baseline/backtest.py` 的一个底层问题：
  - `next_open` 回测只按索引对齐 `benchmark_open`；
  - 没有在进入公共交易日集合前排除 `benchmark_open` 的空值日期；
  - 导致早期样本可能出现 `common_index` 包含日期、但 `benchmark_open` 在该日期已被 `dropna` 掉，随后触发 `KeyError`。
- 已修复为：
  - 先对 `benchmark_open` 做 `dropna()`；
  - 再进入 `common_index` 对齐。
- 这次修复是底层稳健性修复，不改变已有正常样本上的策略逻辑，只避免早期缺口把正式诊断跑断。

### 候选设计
- 基线：
  - `none`
  - `v2`
- 组级 ablation：
  - `ablate_group_volume`
  - `ablate_group_volatility`
  - `ablate_group_structure`
- 因子级 ablation：
  - `ablate_factor_volume_contraction`
  - `ablate_factor_price_volume_divergence`
  - `ablate_factor_volatility_20`
  - `ablate_factor_volatility_contraction`
  - `ablate_factor_close_strength`
  - `ablate_factor_range_position_20`
  - `ablate_factor_drawdown_20`

### 结果一：`v2` 不是“过于简陋”，但内部确实已经出现过重项
- `v2` 基线本轮规则层隔离结果为：
  - 全样本超额 Sharpe 约 `-0.139`
  - 最近完整窗口超额收益约 `-4.09%`，超额 Sharpe 约 `-0.175`
  - 最新弱窗口超额收益约 `+0.08%`
  - `trend_up_low_vol` `20d RankIC` 均值约 `0.180`
- 这再次说明：
  - 当前 `none / v2` 更像规则层锚点，而不是独立主引擎；
  - 但 `v2` 作为锚点内部，已经不是“所有信号都该继续保留”的状态。

### 结果二：`range_position_20 / drawdown_20 / price_volume_divergence` 更像当前应保留的骨架
- 去掉 `range_position_20` 后：
  - 全样本超额 Sharpe 从约 `-0.139` 恶化到约 `-0.345`
  - 最新弱窗口从约 `+0.08%` 降到约 `-3.35%`
- 去掉 `drawdown_20` 后：
  - 全样本超额 Sharpe 恶化到约 `-0.287`
  - 最新弱窗口降到约 `-3.52%`
- 去掉 `price_volume_divergence` 后：
  - 全样本超额 Sharpe 恶化到约 `-0.211`
  - 最新弱窗口降到约 `-7.44%`
- 这说明：
  - `v2` 的“结构位置 + 回撤约束 + 量价背离”仍然更像当前骨架；
  - 下一轮不应优先动这三项。

### 结果三：`volume_contraction / volatility_contraction` 出现“过重嫌疑”
- 去掉 `volume_contraction` 后：
  - 全样本超额 Sharpe 反而升到约 `0.142`
  - 最近完整窗口升到约 `39.44% / 1.106`
  - 最新弱窗口升到约 `31.01%`
  - 但 `trend_up_low_vol` `20d RankIC` 均值从约 `0.180` 小幅回落到约 `0.169`
- 去掉 `volatility_contraction` 后：
  - 全样本超额 Sharpe 升到约 `0.145`
  - 最近完整窗口升到约 `5.76% / 0.271`
  - 最新弱窗口升到约 `3.16%`
  - `trend_up_low_vol` `20d RankIC` 均值升到约 `0.194`
- 这说明：
  - 这两项至少在当前 `v2` 里的权重有“偏重”嫌疑；
  - 其中 `volatility_contraction` 更像值得优先保留为低权重候选；
  - `volume_contraction` 则更像应优先做减权/移除复核的对象。

### 结果四：整组删除过于粗糙，不适合作为下一步
- 去掉整个 `structure` 组后：
  - 全样本超额 Sharpe 反而升到约 `0.111`
  - 但最新弱窗口直接掉到约 `-11.25%`
- 去掉整个 `volume` 组后：
  - 最新弱窗口升到约 `20.24%`
  - 但全样本超额 Sharpe 降到约 `-0.239`
- 去掉整个 `volatility` 组后：
  - 最新弱窗口升到约 `10.30%`
  - 但 `RankIC` 均值显著降到约 `0.109`
- 这说明：
  - 组级改动太粗，容易出现“修一边、坏一边”；
  - 下一轮应坚持小步减法，不做整组删除。

### 本轮结论
1. `none / v2` 这条规则层并不算“过于简陋”，但 `v2` 内部确实已经出现需要清理的过重项。
2. 当前更应保留的骨架是：
   - `range_position_20`
   - `drawdown_20`
   - `price_volume_divergence`
3. 当前最值得进入 `v2.1` 微调首批候选的是：
   - `volume_contraction`
   - `volatility_contraction`
4. 下一轮不做整组删除，也不重开新的 profile 家族；优先做 `v2.1` 小范围减法微调。

### 当前决策
1. 执行端默认值继续保持为：`advanced_ml (ma50 baseline, lgbm) + liquid500 + next_open`
2. `none / v2` 继续保留为规则层锚点，不上升为新的主线替换议题
3. `v2.1` 的首批微调方向固定为：
   - 优先下调或移除 `volume_contraction`
   - 优先下调或移除 `volatility_contraction`
   - 固定保留 `range_position_20 / drawdown_20 / price_volume_divergence`
4. 下一步继续围绕 `v2.1` 做小范围减法微调，不扩 `v3 / v4` 分支

## 2026-03-24 执行端第二十一轮口径复核：`rebalance_freq=5d` 是否仍为真 `5d`
### 本轮目标
- 直接回答当前 `advanced_ml` 主线里的 `rebalance_freq=5d` 究竟代表什么：
  - 它是否仍像早期 `score + 5d` 一样，是真正作用到目标权重上的调仓约束；
  - 还是已经只剩历史参数名，当前实际行为更接近日频目标更新。
- 如果确认口径已经漂移，就把研究端、执行端和文档里的理解重新对齐。

### 本轮产物
- 执行端最小复现实验：
  - `daily_research/output/rebalance_freq_audit_exec/exec_1d_audit`
  - `daily_research/output/rebalance_freq_audit_exec/exec_5d_audit_samepool`
- 研究端最小复现实验：
  - `daily_research/output/rebalance_freq_audit_research_1d`
  - `daily_research/output/rebalance_freq_audit_research_5d`

### 结果一：早期 stage1 基线仍是真 `5d`
- 静态代码复核显示：
  - `daily_research/baseline/run_daily_research.py` 仍保留 `_apply_rebalance_frequency()`
  - 且会在 `build_target_weights()` 之后，把 `target_weights` 与 `score_for_backtest` 一并做频率约束
- 这说明：
  - 早期 `score + 5d` 的研究结论本身没有问题；
  - 但它严格对应的是 stage1 基线路径，不能自动外推到后来的 `advanced_ml` 主线。

### 结果二：当前 `advanced_ml` 研究端不会真正应用 `rebalance_freq`
- 静态代码复核显示：
  - `daily_research/baseline/run_advanced_daily_research.py` 会读取并记录 `rebalance_freq`
  - 但在 `build_target_weights(final_score, ...)` 之后，直接进入后续回测，没有像 stage1 那样再做 `_apply_rebalance_frequency()`
- 最小复现实验显示：
  - `rebalance_freq=1d` 与 `rebalance_freq=5d` 的研究端正式输出里，`metrics / equity_curve / actions / latest_scores / regime_state / training_log / factor_ic_summary / factor_quantile_returns` 全部逐文件一致
  - `metrics.json` 的唯一差异只剩：
    - `rebalance_freq: 1d`
    - `rebalance_freq: 5d`
- 这说明：
  - 当前 `run_advanced_daily_research.py` 的 `rebalance_freq` 已经只剩元数据意义；
  - 它不会改变研究端的真实目标权重路径。

### 结果三：当前执行端也不会因为 `1d / 5d` 改变计划输出
- 静态代码复核显示：
  - `daily_research/baseline/generate_daily_trade_plan.py` 会读取 `rebalance_freq`
  - 但在构建完 `target_weights` 后，会直接取 `target_weights.loc[signal_date]`
  - 中间不存在与 stage1 等价的频率约束步骤
- 最小复现实验显示：
  - 同一模型、同一股票池、同一账户快照下，`rebalance_freq=1d` 与 `rebalance_freq=5d` 的 `actions_today.csv` 完全一致
  - `watchlist.csv` 也完全一致
  - `plan_summary.json` 只有缓存命中元数据不同，不涉及任何计划动作差异
- 这说明：
  - 当前执行端对 `rebalance_freq` 的处理，也已经不再是“真 `5d`”；
  - 当前计划生成行为实质上等价于日频目标更新。

### 本轮结论
1. `advanced_ml` 主线里的 `rebalance_freq=5d` 已经不是早期 stage1 那种“真 `5d` 调仓约束”。
2. 当前研究端与执行端都已确认：`rebalance_freq=1d / 5d` 只改元数据，不改真实目标权重与计划输出。
3. 因此当前执行默认口径 `advanced_ml (ma50 baseline, lgbm) + liquid500 + next_open` 应解释为“日频目标更新”，而不能再直接沿用旧 `score + 5d` 的理解。
4. 这次先不静默改执行语义；下一步若要继续处理这条线，应先做清晰选择：
   - 恢复 `advanced_ml` 主线里的真 `5d` 调仓约束；
   - 或正式把这条主线标准化为日频目标更新，并清理历史文档表述。

### 当前决策
1. 执行端默认值继续保持为：`advanced_ml (ma50 baseline, lgbm) + liquid500 + next_open`
2. 但当前主线的实际调仓语义更新为：按“日频目标更新”理解，而不是“真 `5d`”
3. 早期 `score + 5d` 的研究结论继续保留，但仅限对应 stage1 基线路径，不再直接引用为当前 `advanced_ml` 主线口径
4. 下一步先不改代码行为；等后续再明确是恢复真 `5d`，还是正式标准化为日频目标更新

## 2026-03-24 执行端第二十二轮正式决策：统一当前主线口径
### 本轮目标
- 不再停留在“口径提醒”层面，而是基于当前代码、默认模型产物与已有正式研究记录，把三个容易混淆的问题一次性定下来：
  - 当前默认 `lgbm` 的训练窗口是否需要立即继续拉长；
  - 当前 `advanced_ml` 主线到底按“真 `5d`”还是“日频目标更新”理解；
  - 市场状态过滤与 `none / v2` profile 之后该放在什么层级，不再混成同一条主引擎叙述。

### 结果一：训练窗口保持当前近两年滚动，不机械拉长
- 当前默认模型产物 `daily_research/execution/models/latest_ml_model.json` 对应的真实训练口径显示：
  - 请求起点仍可写为 `2021-01-01`
  - 但有效原始历史窗口约为 `2023-03-06 -> 2026-03-24`
  - 真正用于三组 horizon 训练的样本大致为 `2024-02-22 -> 2026-03-23`
- 这说明：
  - 当前默认 `lgbm` 并不是“吃满 2021 以来全部历史”的长期训练；
  - 它本质上仍是一条近两年滚动、偏近期适应性的主线。
- 因此正式决策为：
  - 当前不机械继续拉长训练窗口；
  - 默认训练口径继续维持 `ml_train_window_days=504`。

### 结果二：验证窗口应继续扩，而不是拿训练窗口替代稳定性判断
- 稳定性问题要靠更长的正式复验窗口回答，而不是靠把训练窗口越喂越长来回答。
- 因此正式决策为：
  - 后续正式复验优先把研究验证起点从 `2021` 往 `2019` 扩；
  - 若数据质量、基准对齐与运行成本允许，再继续评估是否扩到 `2018`。

### 结果三：当前 `advanced_ml` 主线正式标准化为“日频目标更新”
- 承接第二十一轮的静态复核与 `1d / 5d` 最小复现实验，当前已经没有必要继续保留“以后再决定”的模糊态。
- 因此正式决策为：
  - 当前 `advanced_ml` 主线不再按“真 `5d` 调仓约束”理解；
  - 直接标准化为“日频目标更新”；
  - 主入口相关默认值同步统一到 `rebalance_freq=1d`。
- 本轮同步更新的入口包括：
  - `daily_research/baseline/run_advanced_daily_research.py`
  - `daily_research/baseline/generate_daily_trade_plan.py`
  - `daily_research/baseline/compare_ml_model_families.py`
  - `daily_research/baseline/scan_execution_repair_candidates.py`

### 结果四：状态过滤与 `none / v2` 的定位正式收束
- 市场状态过滤这条线继续成立，但只作为门控层，不再被叙述成“主收益引擎”。
- `none / v2` 这条线继续成立，但更适合作为规则层先验，而不是继续扩成 profile zoo。
- 当前真正的主收益引擎继续明确为默认 `lgbm`。

### 本轮结论
1. 你之前的研究主方向没有走偏，但此前确实存在训练窗口、验证窗口、`score + 5d` 与当前主线语义混用的问题。
2. 这些问题现在已正式收束为统一口径：
   - 训练窗口不机械拉长；
   - 验证窗口继续向更长历史扩展；
   - 当前 `advanced_ml` 主线正式按日频目标更新理解。
3. 市场状态过滤与 `none / v2` 继续有效，但层级明确更新为：
   - 状态过滤 = 门控层
   - `none / v2` = 规则层先验
   - `lgbm` = 当前主引擎

### 当前决策
1. 执行端默认值继续保持为：`advanced_ml (ma50 baseline, lgbm) + liquid500 + next_open`
2. 默认训练窗口继续保持近两年滚动，不机械拉长
3. 后续正式研究验证窗口优先从 `2021` 向 `2019` 扩，必要时再评估 `2018`
4. `advanced_ml` 主线正式标准化为“日频目标更新”，不再沿用旧 `score + 5d` 口径
5. 市场状态过滤继续作为门控层，`none / v2` 继续作为规则层先验；当前不把 profile 家族扩成新的执行端主线研究

## 2026-03-24 执行端第二十三轮正式微调：`v2.1` 小范围减法微调
### 本轮目标
- 承接首轮 `v2` 减法诊断，不再做“整组删除”或重开 `v3 / v4`：
  - 只围绕首轮暴露出的两个过重项做小范围减法微调：
    - `volume_contraction`
    - `volatility_contraction`
- 这轮要直接回答三个问题：
  - 对 `v2` 来说，是“单独下调 `volume_contraction`”更有效，还是“单独下调 `volatility_contraction`”更有效；
  - 两者一起减时，是否能比单独减更稳；
  - 有没有已经值得进入下一轮正式归因诊断的 `v2.1` 候选。

### 本轮产物
- 正式输出目录：
  - `daily_research/output/v21_rule_tuning_20260324_formal_round2`
- 工具：
  - `daily_research/tools/v2_rule_ablation_report.py`
  - 本轮新增入口：`--candidate-set v21`
- 候选集合：
  - `v2`
  - `v21_volume_contraction_035`
  - `v21_volume_contraction_025`
  - `v21_volume_contraction_015`
  - `v21_volatility_contraction_020`
  - `v21_volatility_contraction_015`
  - `v21_volatility_contraction_010`
  - `v21_dual_mild_035_020`
  - `v21_dual_balanced_025_015`

### 结果一：单独下调 `volume_contraction` 明显强于单独下调 `volatility_contraction`
- `v21_volume_contraction_015`：
  - 全样本超额 Sharpe 约 `0.095`
  - 最近完整窗口超额收益约 `+21.66%`
  - 最新弱窗口超额收益约 `+22.89%`
  - `trend_up_low_vol` `20d RankIC` 均值约 `0.176`
- 相对基线 `v2`：
  - 全样本超额 Sharpe 提升约 `+0.234`
  - 最近完整窗口超额收益提升约 `+25.75%`
  - 最新弱窗口超额收益提升约 `+22.81%`
  - `RankIC` 只小幅回落约 `-0.004`
- `v21_volume_contraction_025` 也有改善，但明显弱于 `0.15` 档：
  - 全样本超额 Sharpe 仍约 `-0.063`
  - 最近完整窗口超额收益约 `+4.01%`
  - 最新弱窗口超额收益约 `+3.83%`
- 这说明：
  - 当前 `v2` 的主要过重项更像是 `volume_contraction`
  - 而且“适度减一点”还不够，真正有信息量的是更大幅度地下调。

### 结果二：单独下调 `volatility_contraction` 有信息量，但不构成头号方向
- `v21_volatility_contraction_020 / 015 / 010` 的结果都没有跑出像 `volume_contraction_015` 那样的改善：
  - 最好的 `0.10` 档，全样本超额 Sharpe 仍约 `-0.099`
  - 最新弱窗口反而回落到约 `-4.63%`
  - 虽然 `RankIC` 均值有所回升到约 `0.189`
  - 但收益口径没有同步改善
- 这说明：
  - `volatility_contraction` 确实不是毫无问题；
  - 但它更像二级修饰项，而不是这轮最该优先动的主矛盾。

### 结果三：双因子一起减没有跑赢“单独下调 `volume_contraction`”
- `v21_dual_mild_035_020`：
  - 全样本超额 Sharpe 约 `-0.180`
  - 最新弱窗口约 `-1.98%`
- `v21_dual_balanced_025_015`：
  - 全样本超额 Sharpe 约 `-0.269`
  - 最新弱窗口约 `-4.08%`
- 两条双因子候选都不如 `v21_volume_contraction_015`，也不如 `v21_volume_contraction_025`
- 这说明：
  - 首轮看到的两个“过重项”并不意味着这轮应该同步动两把刀；
  - 现在更像是先把 `volume_contraction` 这个主矛盾拆干净，再考虑要不要动第二项。

### 本轮结论
1. `v2.1` 这轮小范围减法微调已经跑出一个非常清晰的头号候选：`v21_volume_contraction_015`。
2. 当前最重要的信息不是“`v2` 里两项都偏重”，而是：
   - `volume_contraction` 的过重问题远比 `volatility_contraction` 更关键；
   - 且应优先用“单独大幅下调”而不是“双因子一起减”来修。
3. 这轮还不足以直接把 `v21_volume_contraction_015` 升为新的执行端规则层默认值，因为它的改善幅度很大，下一步必须先做季度集中度与归因诊断。

### 当前决策
1. 执行端默认值继续保持为：`advanced_ml (ma50 baseline, lgbm) + liquid500 + next_open`
2. `none / v2` 继续保留为规则层先验，不扩 `v3 / v4`
3. 当前 `v2.1` 的头号候选更新为：`v21_volume_contraction_015`
4. `volatility_contraction` 继续保留为次级观察位，但不是下一轮第一优先微调对象
5. 下一步不继续盲扫网格；先对 `v21_volume_contraction_015` 做季度集中度与归因诊断，再决定是否正式晋级为 `v2.1`

## 2026-03-24 执行端第二十四轮正式诊断：`v21_volume_contraction_015` 季度集中度与归因
### 本轮目标
- 不再继续扩 `v2.1` 减权网格，而是先对第二十三轮跑出来的头号候选 `v21_volume_contraction_015` 做最后一步正式诊断：
  - 它相对当前 `v2` 的增量是否已经足够平滑；
  - 还是仍然主要由少数季度、少数月份、少数交易日或少数槽位放大。
- 如果仍偏集中，就按既定规则先不晋级，避免规则层又走回“局部修好就急着替换”的老路。

### 本轮产物
- 正式输出目录：
  - `daily_research/output/v21_rule_concentration_20260324_formal_round3`
- 工具：
  - `daily_research/tools/v21_rule_concentration_report.py`
- 关键文件：
  - `summary_rows.csv`
  - `quarterly_metrics.csv`
  - `candidate_vs_baseline_quarterly_compare.csv`
  - `focus_quarter_daily.csv`
  - `focus_quarter_monthly.csv`
  - `focus_quarter_top_days.csv`
  - `focus_quarter_weight_delta.csv`
  - `focus_quarter_overlap.csv`
  - `quarterly_rankic_compare.csv`
  - `report.json`
  - `report.md`

### 结果一：季度集中度明显好于前面失败分支，但仍不够平滑
- `v21_volume_contraction_015` 相对 `v2`：
  - `19` 个季度里 `9` 个季度更强、`8` 个季度更弱
  - 最佳季度 `2026Q1` 占正向季度总优势约 `34.95%`
  - Top2 季度占比约 `54.16%`
  - Top3 季度占比约 `72.87%`
  - 正向季度 HHI 约 `0.210`
- 这说明：
  - 它确实不像前面很多失败候选那样，单季度就占掉一半以上正向优势；
  - 但离“多季度平滑抬升”的正式晋级标准也还有距离。

### 结果二：焦点季度自动落在 `2026Q1`，且增量主要堆在 `2026-01`
- 自动识别出的焦点季度是 `2026Q1`：
  - compound 超额边际约 `+15.47%`
  - 全季都以 `trend_up_low_vol` 为主，`regime_active_ratio` 约 `0.68`
- 但月度拆解显示：
  - `2026-01` compound 超额边际约 `+13.98%`
  - `2026-02` 仅约 `+0.61%`
  - `2026-03` 仅约 `+0.20%`
- 这说明：
  - 当前这轮增量虽不再是单日孤点，但仍然强烈集中在 `2026Q1` 的前半段，尤其是 `2026-01`。

### 结果三：焦点季度内仍存在少数日期与少数槽位放大
- `2026Q1` 内部：
  - Top5 正向日占正向日总优势约 `61.20%`
  - Top10 正向日占比约 `87.85%`
  - 平均持仓重叠 Jaccard 约 `0.599`
  - `51` 个交易日里，仍有 `23` 天至少与 `v2` 重合 `4` 个名字，占比约 `45.10%`
- 这说明：
  - `v21_volume_contraction_015` 不是整套组合完全改写；
  - 增益更像少数槽位替换叠加少数关键日放大。

### 结果四：收益改善没有伴随季度 RankIC 的同步升级
- `trend_up_low_vol` 的季度 RankIC 对照结果：
  - 仅 `4/17` 个季度强于 `v2`
  - `13/17` 个季度反而更弱
  - 焦点季度 `2026Q1` 本身也落后约 `-0.004`
- 这说明：
  - 这轮收益上的改善，还不能被解释成“规则排序质量已经稳定升级”；
  - 更像是交易路径、权重分布或局部样本结构上的收益放大。

### 本轮结论
1. `v21_volume_contraction_015` 的收益改善是真实的，但当前还不够干净，不满足立即正式晋级为默认 `v2.1` 的条件。
2. 它相对 `v2` 的季度集中度已经比前几条失败分支温和很多，说明这条线不是纯粹无效，值得保留。
3. 但 `2026Q1` 尤其 `2026-01` 的集中仍然偏强，且季度 RankIC 并没有同步改善，因此现在更像“收益候选”，还不是“排序质量已经稳态升级”的候选。
4. 所以当前最稳妥的动作不是立刻晋级，而是继续保留为头号规则层研究候选，先拆解 `2026Q1 / 2026-01` 的关键槽位与交易日来源。

### 当前决策
1. 执行端默认值继续保持为：`advanced_ml (ma50 baseline, lgbm) + liquid500 + next_open`
2. `none / v2` 继续保留为规则层先验，不扩 `v3 / v4`
3. `v21_volume_contraction_015` 继续保留为头号规则层研究候选
4. 但当前暂不正式晋级为默认 `v2.1`
5. 下一步不再扫参数，先拆解 `2026Q1` 尤其 `2026-01` 的关键槽位与交易日来源，确认这轮改善能否抽象成更稳的规则逻辑
## 2026-03-24 执行端第二十五轮正式诊断：`v21` 规则逻辑抽象
### 本轮目标
- 暂时搁置其它研究方向，不再继续扫参数；
- 直接拆解 `v21_volume_contraction_015` 相对 `v2` 在 `2026Q1` 尤其 `2026-01` 的关键交易日与关键槽位来源；
- 回答一个核心问题：
  - 这轮改善能否被抽象成一条可跨季度复放的、更稳的规则逻辑。

### 本轮产物
- 汇总目录：
  - `daily_research/output/v21_rule_logic_20260324_formal_round4`
- 关键工具：
  - `daily_research/tools/v21_rule_logic_report.py`
- 关键文件：
  - `focus_quarter_daily.csv`
  - `focus_month_daily.csv`
  - `focus_quarter_top_days.csv`
  - `focus_month_top_days.csv`
  - `focus_quarter_factor_diff.csv`
  - `focus_month_factor_diff.csv`
  - `selected_logic_factors.csv`
  - `quarter_rank_ic.csv`
  - `rank_ic_summary.csv`
  - `quarter_slot_edge.csv`
  - `slot_edge_summary.csv`
  - `focus_month_stock_signature.csv`
  - `report.json`
  - `report.md`

### 结果一：`2026-01` 的关键槽位来源已经能被描述出来
- 焦点季度自动落在 `2026Q1`，焦点月份自动落在 `2026-01`。
- 关键新增槽位主要集中在：
  - `002716.SZ`
  - `000603.SZ`
  - `688521.SH`
  - `000547.SZ`
  - `603920.SH`
  - `002413.SZ`
  - `002851.SZ`
  - `600219.SH`
- 对应关键减仓槽位主要集中在：
  - `000933.SZ`
  - `600456.SH`
  - `600096.SH`
  - `603063.SH`
  - `600711.SH`
  - `002611.SZ`
  - `601995.SH`
  - `002409.SZ`
  - `600919.SH`
  - `002353.SZ`
- 这说明：
  - `v21` 的改善并不是“整个组合完全重写”；
  - 更像少数槽位在 `2026-01` 里连续被切换到一批更强势、更低波的名字。

### 结果二：可以抽出一组清晰的 signed 因子签名，但它更像槽位偏好而不是稳定排序逻辑
- 从 `2026Q1 / 2026-01` 的关键交易日与关键槽位里抽出的 signed 因子，前 12 个为：
  - `+kama_slope`
  - `+mom_20`
  - `+mom_60`
  - `+trend_slope_20`
  - `+ma_gap_10`
  - `+ma_gap_20_60`
  - `+mom_5`
  - `+price_volume_divergence`
  - `-volatility_20`
  - `+trend_streak`
  - `+up_day_ratio_10`
  - `-atr_14_pct`
- 这说明：
  - `v21` 在焦点月份更偏好“更强趋势 + 更强价量背离 + 更低波动/ATR”的名字；
  - 这套签名是能被清楚描述出来的，不是完全不可解释的随机命中。

### 结果三：这套逻辑能解释槽位偏好，但不能稳定解释未来超额
- `logic_signed` 相对 `v2` 的季度 RankIC：
  - 均值约 `-0.067`
  - 只在 `1` 个季度更强
  - 非焦点季度也只在 `1` 个季度更强
  - 焦点季度 `2026Q1` 本身反而落后 `v2` 约 `-0.157`
- `logic_signed` 相对候选 `v21`：
  - 也只在 `1` 个季度更强
  - 焦点季度落后约 `-0.153`
- 这说明：
  - 抽出来的 signed 逻辑并没有把 `v21` 的改善沉淀成稳定的未来超额排序；
  - 它更像是“解释了候选为什么会买这些股票”，而不是“证明这些股票类型在别的季度也会持续更优”。

### 结果四：槽位复放层面也不够干净
- `logic_signed` 的季度槽位边际：
  - `focus_quarter_slot_edge` 约 `0.528`
  - `positive_slot_edge_quarters = 15`
  - `positive_slot_edge_other_quarters = 14`
  - 但真正与候选正收益季度对齐的只有 `8`
- 这说明：
  - 这套 logic 在很多季度都能解释“候选更喜欢什么类型的名字”；
  - 但这种偏好并不稳定对应更好的未来超额，仍然存在明显的“解释对了偏好，却没有解释对收益”问题。

### 本轮结论
1. `v21_volume_contraction_015` 的 `2026Q1 / 2026-01` 改善已经能被拆成一组比较清晰的槽位偏好签名。
2. 这组签名主要是“更强趋势 + 更强价量背离 + 更低波动/ATR”。
3. 但这套 signed 逻辑目前还不能跨季度稳定复放为更稳的规则排序逻辑。
4. 因此，当前不能把这轮改善视为默认 `v2.1` 的正式逻辑升级。
5. `v21_volume_contraction_015` 继续保留为规则层研究候选，但这条线到这里再次停止晋级，不再继续扫参数，也不继续硬抽新的 profile 默认值。

### 当前决策
1. 执行端默认值继续保持为：`advanced_ml (ma50 baseline, lgbm) + liquid500 + next_open`
2. `none / v2` 继续作为规则层先验，不扩 `v3 / v4`
3. `v21_volume_contraction_015` 保留为研究附录与观察候选，但不正式晋级为默认 `v2.1`
4. 若后续继续规则层研究，优先回到解释 `volume_contraction` 的风险/组合作用，而不是继续追逐 `2026-01` 的局部槽位签名
## 2026-03-24 执行端第二十六轮正式复验：当前默认主线长窗口验证
### 本轮目标
- 按新的研究判断，不再先做“观察期”，而是直接用回测验证当前默认执行主线：
  - `advanced_ml (ma50 baseline, lgbm) + liquid500 + next_open`
- 把正式验证起点从 `2021` 往前扩到 `2019`，回答一个核心问题：
  - 当前默认主线是否只是“近期窗口好看”，还是在更长历史里仍然成立。

### 本轮过程
1. 先直接运行 `2019` 起的正式复验。
2. 首次运行时，暴露出一个长窗口真实 bug：
   - `daily_research/baseline/ml_alpha.py` 在 `train_regime_only` 口径下，默认假设 `regime_state` 与 `label_df` 的日期完全对齐；
   - 当请求窗口拉长后，`2019-01-02` 这类更早日期会直接触发 `KeyError`。
3. 该问题已修复为：
   - 先按 `label_df.index` 对 `regime_state["regime_on"]` 做安全重索引；
   - 再执行 regime-only 训练过滤；
   - 短窗口语义不变，长窗口不再因早期状态缺口崩溃。

### 本轮产物
- 正式输出目录：
  - `daily_research/output/advanced_ml_ma50_lgbm_longwindow_2019_formal`
- 补充汇总：
  - `daily_research/output/advanced_ml_ma50_lgbm_longwindow_2019_formal/long_window_review/window_review_summary.json`
  - `daily_research/output/advanced_ml_ma50_lgbm_longwindow_2019_formal/long_window_review/run_summary_rows.csv`
  - `daily_research/output/advanced_ml_ma50_lgbm_longwindow_2019_formal/long_window_review/window_slice_summary.csv`
- 关键修复文件：
  - `daily_research/baseline/ml_alpha.py`

### 结果一：`2019` 起长窗口下，当前默认主线仍然成立
- `2019` 起正式复验结果：
  - 全样本超额收益约 `529.30%`
  - 全样本超额 Sharpe 约 `1.578`
  - 全样本超额最大回撤约 `-27.50%`
  - 平均换手约 `0.497`
  - `regime_active_ratio` 约 `0.449`
- 这说明：
  - 新增更早历史后，主线明显变难；
  - 但没有失效，仍保持正超额、正 Sharpe、可接受回撤。

### 结果二：相对 `2021` 口径，长窗口确实变弱，但不是塌掉
- 当前 `2021` 口径的参考结果为：
  - 全样本超额收益约 `887.75%`
  - 全样本超额 Sharpe 约 `2.300`
  - 全样本超额最大回撤约 `-23.43%`
- 与之相比，`2019` 起版本：
  - 超额 Sharpe 从约 `2.300` 回落到约 `1.578`
  - 超额最大回撤从约 `-23.43%` 扩到约 `-27.50%`
- 这说明：
  - 更早窗口确实更难做；
  - 但当前默认主线并没有因为把验证口径拉长就被推翻。

### 结果三：这轮还确认了一个更关键的口径边界
- “请求起点 = 2019” 并不等于 “真实交易样本从 2019 开始”。
- 在当前 `ml_train_window_days=504` 与滚动 liquid500 研究池口径下，本轮 `2019` 复验的：
  - `first_nonzero_pool_date = 2019-01-31`
  - `first_equity_date = 2021-08-02`
  - `first_holding_date = 2021-10-20`
- 作为对照，原 `2021` 口径的：
  - `first_equity_date = 2022-01-04`
  - `first_holding_date = 2022-06-01`
- 这说明：
  - 把请求起点从 `2021` 推到 `2019`，确实把真实交易样本往前拉了约 `7` 个多月；
  - 但 `2019 ~ 2021` 里仍有很大一段在承担训练/预热作用，而不是完整交易样本。

### 本轮结论
1. 当前默认执行主线在更长历史里仍然成立，不支持“这条主线只是近期窗口有效”的判断。
2. 相对 `2021` 口径，长窗口确实明显变弱，但仍保持正超额、正 Sharpe，因此默认主线不需要因为这轮复验而回滚。
3. 这轮更重要的新增认识是：当前研究框架下，“请求起点”与“真实交易起点”不是一回事；长训练窗和滚动股票池预热会显著吞掉更早历史。
4. 因此，如果后续真要把真实交易样本再往前压，不应继续机械把 start-date 从 `2019` 改到 `2018`，而应先解决更早可用数据边界和长训练窗预热问题。

### 当前决策
1. 执行端默认值继续保持为：`advanced_ml (ma50 baseline, lgbm) + liquid500 + next_open`
2. 不再把“先观察一段时间”作为进入下一研究阶段的前置动作
3. 下一步正式转向 `deep_alpha` 的最终判决
4. 若后续还要继续做更早历史的执行端正式复验，优先级高于“继续往 `2018` 改 start-date`”的，是先补更早可用数据或重新设计长训练窗预热口径
## 2026-03-24 执行端第二十七轮正式诊断：连续状态分数首轮验证
### 本轮目标
- 保留现有四象限门控不动；
- 不直接改默认执行逻辑；
- 先验证一层“宽度 + 分歧 + 流动性”的连续状态分数，是否能解释 `regime_on` 内部的弱窗口与错误开仓。

### 本轮产物
- 汇总目录：
  - `daily_research/output/continuous_market_context_20260324_formal_round1`
- 关键工具：
  - `daily_research/baseline/market_context.py`
  - `daily_research/tools/continuous_market_context_report.py`
- 关键文件：
  - `context_features.csv`
  - `decision_context_diagnostic.csv`
  - `regime_on_context_diagnostic.csv`
  - `regime_on_quantile_summary.csv`
  - `regime_on_quadrant_summary.csv`
  - `context_correlation_summary.csv`
  - `latest_weak_window_summary.csv`
  - `top_wrong_open_days.csv`
  - `report.json`
  - `report.md`

### 结果一：连续状态分数对后续 20 日市场环境已有解释力
- 在当前默认主线 `advanced_ml (ma50 baseline, lgbm) + liquid500 + next_open` 上，`regime_on` 样本按 `context_score` 做五分位后：
  - Q1 的 `benchmark_fwd_20d` 均值约 `0.55%`
  - Q5 的 `benchmark_fwd_20d` 均值约 `2.47%`
- 同时，`wrong_open_rate_20d` 也明显分层：
  - Q1 约 `50.00%`
  - Q5 约 `29.41%`
- `context_score` 与 `benchmark_fwd_20d` 的相关性也已转正：
  - Pearson 约 `0.226`
  - Spearman 约 `0.278`
- 这说明：
  - 这层分数已经能在“门开了”的前提下，区分哪些日期后面的市场环境更舒服、哪些日期更容易变成错误开仓。

### 结果二：它还没有直接解释成更强的 1 日超额
- `context_score` 与当前主线的 `portfolio_return / excess_return` 相关性都接近于零：
  - `portfolio_return` Pearson 约 `0.014`
  - `excess_return` Pearson 约 `0.017`
- 五分位比较里，Q5 并没有稳定跑赢 Q1 的 `avg_excess_return_1d`：
  - Q1 约 `0.85%`
  - Q5 约 `0.63%`
- 这说明：
  - 这层连续状态分数当前更像“风险与执行调节器”；
  - 还不是能直接拿来替代当前选股 alpha 的东西。

### 结果三：最新弱窗口里低状态分数显著堆积
- 最新弱窗口 `2025-09-05 -> 2026-03-19` 的 `regime_on` 日期：
  - 平均 `context_score` 约 `0.474`
  - 底部五分位占比约 `34.83%`
  - 顶部五分位占比为 `0%`
- 其余 `regime_on` 日期：
  - 平均 `context_score` 约 `0.717`
  - 底部五分位占比约 `3.75%`
  - 顶部五分位占比约 `42.50%`
- 这说明：
  - 当前默认主线的弱窗口，不只是收益结果变差；
  - 连续状态层也已经明确显示出“门虽开着，但开得不够舒服”的结构特征。

### 结果四：当前 `regime_on` 样本几乎全部落在 `trend_up_low_vol`
- 本轮正式样本里，`regime_on` 主体都落在 `trend_up_low_vol`；
- 所以这轮连续状态诊断本质上是在回答：
  - 同样都是低波上涨，哪些日期值得更积极，哪些日期应该收一点。

### 本轮结论
1. 连续状态分数已经表现出明确的“风险门内再分层”价值。
2. 它能解释一部分后续 20 日市场环境与错误开仓风险。
3. 但它还不能直接解释成更强的 1 日超额，因此当前不该把它当成新的 alpha 主引擎。
4. 正确的升级方向应是：保留现有四象限门控不动，只在 `regime_on` 内部拿这层分数去做轻度执行去风险。

### 当前决策
1. 执行端默认值继续保持为：`advanced_ml (ma50 baseline, lgbm) + liquid500 + next_open`
2. 四象限继续作为第一层安全门
3. 连续状态分数先定位为“风险门内再分层”，不直接替换门控，也不直接接入 ML 主引擎
4. 下一步正式进入第二轮：连续状态分数驱动的软调节回测
5. 首批只试：
   - `holding_count`
   - `max_weight`
   - `turnover_limit`
   - `max_style_weight`
6. 若出现“弱窗口改善但强窗口或全样本被破坏”，则停止这条线晋级执行层

## 2026-03-24 执行端第二十八轮正式回测：连续状态分数驱动的软调节
### 本轮目标
- 保留现有四象限门控不动；
- 只在低 `context_score` 的 `regime_on` 日期做轻度去风险；
- 首批单独测试 4 个执行旋钮：
  - `holding_count`
  - `max_weight`
  - `turnover_limit`
  - `max_style_weight`
- 严格执行既定停止规则：
  - 只要出现“弱窗口改善但强窗口或全样本被破坏”，就停止这条线晋级执行层。

### 本轮产物
- 汇总目录：
  - `daily_research/output/continuous_market_context_soft_20260324_formal_round2_defaultwindow`
- 关键工具：
  - `daily_research/tools/continuous_market_context_soft_scan.py`
- 关键文件：
  - `soft_scan_summary.csv`
  - `report.json`
  - `report.md`
  - 各候选子目录下的：
    - `equity_curve.csv`
    - `actions.csv`
    - `target_weights.csv`
    - `metrics.json`
    - `metrics_recent_full.json`
    - `metrics_latest_weak.json`

### 结果一：四个首批候选都没有通过正式停止规则
- `baseline`：
  - 全样本超额 Sharpe 约 `1.601`
  - 最近完整窗口超额 Sharpe 约 `1.416`
  - 最新弱窗口超额收益约 `-26.18%`
  - 最新弱窗口超额 Sharpe 约 `-1.088`
- `low_ctx_hold3`：
  - 全样本超额 Sharpe 回落到约 `1.564`
  - 最近完整窗口超额 Sharpe 回落到约 `1.218`
  - 最新弱窗口恶化到约 `-31.87% / -1.437`
  - 结论：整体退化，不保留
- `low_ctx_maxw20`：
  - 全样本超额 Sharpe 回落到约 `1.466`
  - 最近完整窗口超额 Sharpe 回落到约 `1.096`
  - 最新弱窗口恶化到约 `-30.92% / -1.296`
  - 结论：整体退化，不保留
- `low_ctx_turnover1`：
  - 最近完整窗口超额 Sharpe 提升到约 `1.576`
  - 但全样本超额 Sharpe 回落到约 `1.570`
  - 最新弱窗口超额收益小幅改善到约 `-25.50%`，但 Sharpe 反而微幅回落到约 `-1.101`
  - 结论：不满足“弱窗口与全样本同步改善”，不晋级
- `low_ctx_style40`：
  - 最新弱窗口从约 `-26.18% / -1.088` 小幅改善到约 `-25.97% / -1.081`
  - 但全样本超额 Sharpe 从约 `1.601` 回落到约 `1.593`
  - 最近完整窗口超额 Sharpe 从约 `1.416` 回落到约 `1.395`
  - 结论：触发既定停止规则，`candidate_verdict = stop_due_to_tradeoff`

### 结果二：低状态分数触发本身不是空信号
- 本轮所有候选共用同一批低状态分数触发日：
  - `override_signal_days = 95`
  - `override_signal_ratio = 9.31%`
  - 最新弱窗口里的 `override_ratio` 约 `32.54%`
- 这说明：
  - 连续状态分数确实识别到了不少“门开着但环境不够舒服”的日期；
  - 当前问题不在“触发条件太稀或太假”，而在第一批单旋钮软调节还不够干净。

### 本轮结论
1. 连续状态分数作为诊断层与风险附录的价值继续成立。
2. 但在当前默认窗口下，首批四个单旋钮软调节都没有通过“弱窗口改善且强窗口/全样本不被破坏”的正式晋级门槛。
3. 因此这条连续状态软调节线到这里停止，不进入执行层默认逻辑。
4. 当前默认执行口径继续保持为：`advanced_ml (ma50 baseline, lgbm) + liquid500 + next_open`。

### 当前决策
1. 执行端默认值继续保持为：`advanced_ml (ma50 baseline, lgbm) + liquid500 + next_open`
2. 四象限继续作为第一层安全门
3. 连续状态分数继续保留为诊断层与风险附录，不晋级为执行层软调节逻辑
4. 这条连续状态软调节线到此停止，不继续扩单旋钮参数网格
5. 下一步研究重心回到 `deep_alpha` 的最终正式判决

## 2026-03-24 项目治理回收：README / 计划去日志化
### 本轮目标
- 把 `daily_research/README.md` 收回到“当前状态与入口”；
- 把 `daily_research/daily_research_plan.md` 收回到“当前默认决策与优先级”；
- 让 `research_log.md` 继续作为唯一时间顺序记录；
- 给文档维护补上结构守卫，而不只是靠人工记忆。

### 本轮动作
- 重写 `daily_research/README.md`：
  - 只保留当前默认执行口径、目录职责、日常入口与维护命令；
- 重写 `daily_research/daily_research_plan.md`：
  - 只保留当前默认主线、当前优先级、已降级分支与停止规则；
- 更新根目录 `README.md`：
  - 把当前执行主线写精确为 `advanced_ml (ma50 baseline, lgbm) + liquid500 + next_open`；
  - 补上 `2026-03-23 ~ 2026-03-24` 的主线收束阶段；
- 增强 `daily_research/tools/doc_guard.py`：
  - 默认纳入根目录 `README.md`；
  - 增加 README / 计划文件的行数上限检查；
  - 增加按日期追加日志式标题的结构检查。

### 本轮结论
1. 文档治理已经从“靠人工记住边界”升级为“文档分工 + 守卫脚本”。
2. `README.md` 与 `daily_research_plan.md` 不再承担历史追加职责，完整时间线统一留在 `research_log.md`。
3. 当前更准确的项目状态已经固化为：
   - 执行端默认口径：`advanced_ml (ma50 baseline, lgbm) + liquid500 + next_open`
   - 调仓语义：日频目标更新
   - 当前主研究事项：`deep_alpha` 最终正式判决

### 当前决策
1. 以后新增实验结论继续写入 `research_log.md`，不再顺手堆进 README 或计划文件。
2. 日常维护时先跑 `python daily_research/tools/doc_guard.py check`，再提交文档更新。
3. 继续保留 `workspace_maintenance.py` 作为热区体检与清理入口，不回到人工记忆式维护。

## 2026-03-24 下一阶段正式起步：执行 README 收敛 + `deep_alpha` 口径统一
### 本轮目标
- 把 `daily_research/execution/README.md` 收敛成面向日常操作的执行手册；
- 正式启动 `deep_alpha` 最终判决阶段的第一项治理动作；
- 先统一 `deep_alpha` 的调仓语义默认值，避免后续研究结论继续混入旧 `5d` 元数据。

### 本轮动作
- 重写 `daily_research/execution/README.md`：
  - 把内容收敛为“当前默认口径 + 每日标准流程 + 输出解释 + 风险边界”；
  - 减少历史叙述和分散说明，让执行入口更直接。
- 更新 `daily_research/deep_alpha/config.py`：
  - 默认 `rebalance_freq` 从 `5d` 改为 `1d`。
- 更新 `daily_research/deep_alpha/run_deep_alpha_research.py`：
  - 命令行默认 `--rebalance-freq` 从 `5d` 改为 `1d`；
  - `metrics.json` 显式写出 `rebalance_freq`。
- 更新 `daily_research/README.md`：
  - `deep_alpha` 主线示例命令显式补上 `--rebalance-freq 1d`。

### 本轮结论
1. 执行端说明文档已经从“信息堆叠”收敛为“可直接照着跑的操作手册”。
2. `deep_alpha` 的默认调仓语义已经与当前主线统一到“日频目标更新”。
3. 这标志着 `deep_alpha` 最终正式判决已经从“规划阶段”进入“口径统一后的正式执行阶段”。

### 当前决策
1. 后续 `deep_alpha` 正式对照实验默认按 `rebalance_freq=1d` 记录与解释。
2. 下一步直接进入 `deep_alpha` 当前主线的最小充分对照矩阵，不再先做额外支线扩展。

## 2026-03-24 `deep_alpha` 最小充分对照矩阵：runner 固化与正式起步
### 本轮目标
- 不再手工拼接 `deep_alpha` 正式对照命令；
- 把“最小充分对照矩阵”固化成正式 runner；
- 先从 `backbone` 阶段进入当前主线正式判决。

### 本轮动作
- 新增 `daily_research/deep_alpha/run_minimal_matrix.py`：
  - 自动按最新已完成交易日切最近三段正式 walk-forward 窗口；
  - 当前分三阶段组织最小矩阵：
    - `backbone`
    - `score_head`
    - `ranking`
  - `backbone` 阶段当前只比较：
    - `gru + manual + plain`
    - `patch_transformer + manual + plain`
    - `patch_transformer + masked pretrain + manual + plain`
  - 同一套 runner 会把：
    - `matrix_plan.json`
    - `windows.csv`
    - 各阶段命令清单
    - 各阶段运行汇总 / 选中 recipe
    统一写到同一个输出根目录下。
- 用 `quant` 环境做了 dry-run：
  - 根目录：`daily_research/output/deep_alpha_minimal_matrix_20260324_formal_round1`
  - 生成了：
    - `matrix_plan.json`
    - `windows.csv`
    - `stage_backbone_commands.txt`
- 同时补了主 README 入口：
  - `python daily_research/deep_alpha/run_minimal_matrix.py --phase backbone --root-tag deep_alpha_minimal_matrix_round1`

### 当前 dry-run 切出的正式窗口
1. `2023-02-09 -> 2024-02-22`
2. `2024-02-23 -> 2025-03-10`
3. `2025-03-11 -> 2026-03-24`

### 本轮结论
1. `deep_alpha` 当前主线的最小充分对照矩阵已经从“文字规划”变成“可执行正式入口”。
2. 当前正式起步点已经明确为 `backbone` 阶段，而不是继续横向扩更多研究支线。
3. 后续 `score_head` 与 `ranking` 阶段将建立在 `backbone` winner 之上，而不是直接暴力全因子扩表。

### 当前决策
1. 正式矩阵入口固定为：
   - `daily_research/deep_alpha/run_minimal_matrix.py`
2. 当前第一阶段固定为：
   - `--phase backbone`
3. `score_head` 与 `ranking` 只在上阶段 winner 产生后继续推进。

## 2026-03-24 运行环境基线固化：解释器与调用口径留档
### 本轮目标
- 把当前工作区真实可用的运行环境固定成项目文档；
- 避免后续研究再次混用 `base` 与 `quant` 解释器；
- 给 `deep_alpha` 主线与最小充分对照矩阵保留统一调用前缀。

### 本轮动作
- 新增 `daily_research/runtime_environment.md`：
  - 记录工作区根目录、Shell、时区与环境快照日期；
  - 记录默认 `python` 与推荐 `quant` Python 的路径和版本；
  - 记录 `quant` 环境中的关键依赖版本：
    - `pandas==2.3.3`
    - `numpy==2.0.2`
    - `torch==2.8.0+cpu`
  - 记录可用 conda 环境清单与推荐调用方式。
- 更新 `daily_research/README.md`：
  - 在文档分工里加入 `runtime_environment.md`；
  - 在推荐阅读顺序里补上环境基线入口。

### 本轮结论
1. 当前运行口径已经从“靠会话记忆”固化为项目文档。
2. `daily_research` 的正式研究与 `deep_alpha` 相关脚本默认应使用 `quant` 环境，而不是 `base` Python。
3. 后续如果解释器或关键依赖升级，应先更新环境基线文档，再推进新的正式实验批次。

### 当前决策
1. 正式研究脚本默认调用：
   - `C:\Users\ASUS\miniconda3\envs\quant\python.exe`
2. `base` 环境只视为轻量维护入口，不再假定具备完整研究依赖。

## 2026-03-25 执行端完整四象限复核：坏市场盈利能力与坏状态专用 profile 排查
### 本轮目标
- 直接回答当前执行默认主线在完整四象限里的盈利能力；
- 明确坏市场里当前策略是“绝对赚钱”还是“少亏 / 空仓”；
- 判断当前是否已经存在类似 `v2`、但只在坏市场状态起作用的成熟分支。

### 本轮动作
- 复核当前最贴近执行默认口径的正式产物：
  - `daily_research/output/advanced_ml_model_family_compare_20260323_formal_ma50_execution/lgbm`
  - `daily_research/output/advanced_ml_ma50_lgbm_longwindow_2019_formal`
- 新增输出目录：
  - `daily_research/output/execution_quadrant_review_20260325_formal_round1`
  - 其中生成：
    - `current_execution_lgbm_quadrant_summary.csv`
    - `longwindow_lgbm_quadrant_summary.csv`
    - `bad_market_summary.csv`
    - `histgb_vs_lgbm/`
- 用现有归因脚本补跑：
  - `histgb vs lgbm` 的完整四象限对照与季度对照。
- 复核 `daily_research/baseline/state_profiles.py` 与近几轮正式诊断记录，确认当前已注册与已保留候选主要覆盖哪些状态。

### 结果一：当前默认执行端在坏市场里更像“去风险”，不是稳定绝对盈利
- 当前默认等价正式口径 `current_execution_lgbm`：
  - `trend_down_high_vol`
    - 组合收益约 `0.00%`
    - 超额收益约 `+3.79%`
    - 平均持仓数约 `0.00`
  - `trend_down_low_vol`
    - 组合收益约 `-19.17%`
    - 基准收益约 `-54.14%`
    - 超额收益约 `+66.79%`
    - 平均持仓数约 `0.23`
- 长窗口附录 `longwindow_lgbm` 也保持同一结构：
  - `trend_down_high_vol` 约 `0.00% / -0.98%`（组合 / 超额）
  - `trend_down_low_vol` 约 `-38.11% / +45.01%`
- 这说明：
  - 当前主线在坏市场里的核心能力是空仓、降仓、少亏；
  - 不是已经具备“坏市场里稳定做出绝对正收益”的独立 alpha。

### 结果二：`lgbm` 对坏市场的改进存在，但主要仍是防守改进
- 相对旧 `histgb`：
  - `trend_down_high_vol` 的超额边际约 `+0.00%`
  - `trend_down_low_vol` 的超额边际约 `+5.94%`
  - `trend_up_high_vol` 的超额边际约 `+15.78%`
  - `trend_up_low_vol` 的超额边际约 `+164.71%`
- 这说明 `lgbm` 对坏市场不是完全没改善；
- 但当前默认主线的决定性增益仍主要来自两个上涨象限，尤其 `trend_up_low_vol`。

### 结果三：当前没有成熟的“坏市场专用 `v2` 类 profile”
- `daily_research/baseline/state_profiles.py` 当前正式注册的 profile 只有：
  - `up_low_breakout_v1 / v2 / v3`
  - `up_dual_v1 / v2`
- 它们只覆盖：
  - `trend_up_low_vol`
  - `trend_up_high_vol`
- 当前没有一个已注册、已验证、只在 `trend_down_low_vol / trend_down_high_vol` 生效的正式 profile。
- 历史上确实出现过“弱窗口和 `trend_down_low_vol` 一起改善”的候选，例如：
  - `up_low_ml55_none25_v220` 曾把 `trend_down_low_vol` 的弱窗口超额从约 `-8.14%` 改善到约 `-1.05%`
- 但那类改善的来源是：
  - 降低 `trend_up_low_vol` 下 `ML` 主导权；
  - 不是构建了一个真正独立的坏市场状态 alpha。
- 后续第二轮保留下来的候选 `up_low_ml62_none23_v215 / up_low_ml61_none24_v215` 也已确认：
  - 差异几乎全部来自 `trend_up_low_vol`
  - 两个下跌象限基本没变
  - 因此没有资格被解释成“坏市场专用分支”。

### 本轮结论
1. 当前执行默认主线在坏市场里具备明显防守价值，但还不能说“坏市场本身能稳定盈利”。
2. 当前没有成熟的、已正式验证的“坏市场专用 `v2` 类 profile”。
3. 如果下一步继续研究坏市场增量，正确方向应是：
   - 单独定义 `trend_down_low_vol / trend_down_high_vol` 的防守或反向候选；
   - 而不是继续假定上涨态 profile 的外推会自动修复坏市场。

### 当前决策
1. 执行端默认值继续保持为：`advanced_ml (ma50 baseline, lgbm) + liquid500 + next_open`
2. 四象限门控继续作为第一层安全门，不因为这轮检查而撤掉。
3. 若继续推进“坏市场专项研究”，应明确把目标定义为：
   - 坏市场绝对收益
   - 或坏市场进一步降损
   二者需要分开立题与验收。

## 2026-03-25 坏市场专项目标固定：从“坏市场超额”切到“坏市场绝对收益”
### 本轮目标
- 把“坏市场绝对收益”从口头选择固定成正式研究目标；
- 让后续候选扫描直接按这个目标排序，而不是继续默认按全样本超额 Sharpe；
- 避免后续再次把“坏市场少亏”表述成“坏市场盈利”。

### 本轮动作
- 更新 `daily_research/daily_research_plan.md`：
  - 明确坏市场专项若重启，主目标固定为“坏市场绝对收益”；
  - 明确坏市场专项的停止规则不能只看坏市场超额。
- 更新 `daily_research/baseline/scan_execution_repair_candidates.py`：
  - 新增 `--selection-objective`
    - `default_excess`
    - `bad_market_absolute`
  - 新增 `--bad-market-quadrants`
  - 新增 `--primary-bad-market-quadrant`
  - 候选汇总新增：
    - `full_bad_market_total_return`
    - `full_primary_bad_market_total_return`
    - 各窗口 `bad_market_total_return`
    - 各窗口 `primary_bad_market_total_return`
  - 正式扫描产物新增坏市场专项指标文件：
    - `metrics_bad_market_full.json`
    - `metrics_primary_bad_market_full.json`
    - `metrics_<window>_bad_market.json`
    - `metrics_<window>_primary_bad_market.json`
  - 当目标切到 `bad_market_absolute` 时，最终 summary 改按坏市场绝对收益优先排序。
- 立即对现有正式候选库存做了一次坏市场目标重排：
  - 输出目录：`daily_research/output/bad_market_objective_existing_candidates_20260325`
  - 复核范围：
    - `advanced_ml_execution_repair_scan_20260323_formal_round13_ma50_state_ensemble_round2_low_only`
    - `advanced_ml_execution_repair_scan_20260322_formal_round12_ma50_regime_vol`

### 本轮结论
1. 后续“坏市场专项研究”终于有了清晰目标函数，不再和“坏市场超额修复”混为一谈。
2. 这一步还没有证明我们已经找到坏市场绝对盈利方案，但已经把研究入口改成会朝这个方向收敛。
3. 现有正式候选库存按坏市场绝对收益重排后，当前第一名仍是 `baseline`，且 `positive_bad_market_candidate_count = 0`。
4. 这说明：
   - 当前库存里还没有一个候选已经达到“坏市场绝对盈利”；
   - 下一步必须显式设计坏市场候选，而不是继续指望现有上涨态微调自然外溢成坏市场盈利方案。

### 当前决策
1. 坏市场专项默认目标固定为：`bad_market_absolute`
2. 若后续继续执行端候选扫描，优先显式传：
   - `--selection-objective bad_market_absolute`
3. 若候选只改善坏市场超额、不改善坏市场绝对收益，则只记为防守修复，不记为坏市场盈利方案。

## 2026-03-25 坏市场专项首批专用候选：formal round1 完整收口
### 本轮目标
- 不再继续拿上涨态微调外推坏市场；
- 直接设计面向 `trend_down_low_vol / trend_down_high_vol` 的第一批专用候选；
- 按“坏市场绝对收益”完成首批正式扫描，并判断是否存在值得继续细化的坏市场防守分支。

### 本轮动作
- 更新 `daily_research/baseline/state_profiles.py`：
  - 新增 `upv2_downlow_rebound_v1`
  - 新增 `upv2_downdual_reversal_v1`
- 更新 `daily_research/baseline/scan_execution_repair_candidates.py`：
  - 新增 `bad_market_round1` 候选集；
  - 支持 `--candidate-labels`，允许长时扫描按标签分批续跑；
  - 修正坏市场切片指标口径：
    - 由“直接截取累计 equity”改为“按入选日期自身收益序列重建 equity”；
    - 避免坏市场切片收益把中间非坏市场日期也错误算入。
- 首批候选正式分批产出：
  - `daily_research/output/advanced_ml_bad_market_round1_20260325_formal_round1`
  - `daily_research/output/advanced_ml_bad_market_round1_20260325_formal_round1_core3`
  - `daily_research/output/advanced_ml_bad_market_round1_20260325_formal_round1_badonly_hold2`
  - `daily_research/output/advanced_ml_bad_market_round1_20260325_formal_round1_hold2_remaining`
- 汇总目录：
  - `daily_research/output/advanced_ml_bad_market_round1_20260325_consolidated`
  - 其中生成：
    - `combined_bad_market_round1_summary.csv`
    - `summary.txt`

### 首批正式候选
1. `baseline`
2. `downlow_rebound_open`
3. `downlow_rebound_open_hold2`
4. `downlow_ruleheavy_open`
5. `downdual_reversal_open`
6. `downdual_reversal_open_hold2`
7. `downdual_reversal_badonly`
8. `downdual_reversal_badonly_hold2`

### 汇总结果
- 全历史坏市场绝对收益仍然最好的是 `baseline`：
  - `full_bad_market_total_return = -28.46%`
- 最近弱窗口坏市场绝对收益最强的前三名是：
  1. `downdual_reversal_open_hold2`
     - `latest_weak_bad_market_total_return = -4.72%`
  2. `downdual_reversal_badonly_hold2`
     - `latest_weak_bad_market_total_return = -5.53%`
  3. `downlow_ruleheavy_open`
     - `latest_weak_bad_market_total_return = -6.16%`
- 当前首批候选里仍然没有一个达到坏市场绝对盈利：
  - `positive_full_bad_market_candidate_count = 0`
- `hold2` 在这轮里普遍优于对应 open 母体：
  - `downlow_rebound_open_hold2` 明显优于 `downlow_rebound_open`
  - `downdual_reversal_open_hold2` 明显优于 `downdual_reversal_open`
  - `downdual_reversal_badonly_hold2` 明显优于 `downdual_reversal_badonly`
- 但它们当前改善的主要是：
  - 最近坏市场窗口的少亏与超额修复；
  - 还不是跨长历史稳定成立的坏市场绝对盈利。

### 本轮结论
1. 坏市场专项首批专用候选已经从“口头方向”变成了正式可复核的一轮产物。
2. 这轮没有找到可晋级执行端的坏市场绝对盈利方案，`baseline` 仍是全历史坏市场绝对收益的最优参考。
3. 但近期弱市里已经筛出两个值得继续细化的防守修复方向：
   - `downdual_reversal_open_hold2`
   - `downdual_reversal_badonly_hold2`
4. 这两个方向当前只能记为“近期坏市场防守修复候选”，不能记为“坏市场盈利方案”。

### 当前决策
1. 首批坏市场专用候选不晋级执行端默认值。
2. 若继续推进坏市场专项，下一轮只围绕以下两条线做小步细化：
   - `downdual_reversal_open_hold2`
   - `downdual_reversal_badonly_hold2`
3. 下一轮只允许细化：
   - 持有约束
   - 止盈止损
   - 坏市场专用 `state_horizon_weights / state_ensemble_weights`
4. 在出现“全历史坏市场绝对收益仍显著差于 baseline”时，不再继续横向扩更多 recipe。
## 2026-03-26 `advanced_ml (ma50 baseline, lgbm)` 组合策略消融 / `trend_up_low_vol` state-only / overlap 诊断
### 本轮目标
- 不再只凭经验判断 `ma50 baseline + lgbm + none + v2` 是否合理；
- 正式补齐三类可复现诊断：
  - `ablation`: `ml only / none only / v2 only / ml+none / ml+v2 / ml+none+v2`
  - `state-only`: 只在 `trend_up_low_vol` 扫 `ml:none:v2`
  - `overlap`: 比较 `ml_score` 与 `score_none / score_v2` 的截面相关、Top5 重合率、弱窗口信号增益

### 本轮动作
- 新增统一诊断脚本：
  - `daily_research/baseline/diagnose_advanced_ml_ensemble.py`
- 正式输出目录：
  - `daily_research/output/advanced_ml_ensemble_diag_20260326_formal_rerun`
- 这轮实际数据日期：
  - `latest_data_date = 2026-03-25`
- 这轮因 `--auto-trim-history` 实际使用的训练/评估历史窗口：
  - `2023-03-07 -> 2026-03-25`
- 本轮弱窗口统一按绝对日期命名为：
  - `weak_window_20250905_20260319 = 2025-09-05 -> 2026-03-19`
- 顺手补了共享 `ML per_horizon scores` 缓存，避免以后改权重表时重复训练 30 分钟：
  - `daily_research/cache/advanced_ml/ml_scores/6600fbb205abd471d5ef.pkl`

### 结果一：全局默认三层融合并不是当前最优主组合
- `ml_only`
  - `full_excess_sharpe = 0.379`
  - `full_excess_total_return = +37.02%`
- 当前默认 `ml+none+v2`（全局 `0.70 / 0.20 / 0.10`）
  - `full_excess_sharpe = 0.265`
  - `full_excess_total_return = +23.87%`
- 这说明：
  - 全样本上，当前默认三层融合相对 `ml_only` 已出现明显拖累；
  - 拖累不是来自 `ma50` 边界，而是来自全局 ensemble 结构本身。

### 结果二：弱窗口里真正拖累默认策略的不是 `ma50`，而是“ML 权重过高”
- 在弱窗口 `2025-09-05 -> 2026-03-19` 内：
  - `v2_only`
    - `weak_window_20250905_20260319_excess_sharpe = 0.170`
    - `weak_window_20250905_20260319_excess_total_return = +2.27%`
  - `none_only`
    - `weak_window_20250905_20260319_excess_sharpe = -0.363`
    - `weak_window_20250905_20260319_excess_total_return = -4.42%`
  - `ml_only`
    - `weak_window_20250905_20260319_excess_sharpe = -0.581`
    - `weak_window_20250905_20260319_excess_total_return = -12.35%`
  - 默认 `ml+none+v2`
    - `weak_window_20250905_20260319_excess_sharpe = -0.627`
    - `weak_window_20250905_20260319_excess_total_return = -12.72%`
- 这说明：
  - 弱窗口里，`v2` 规则层比 `ML` 更抗压；
  - 默认组合不仅没有对冲 `ML` 的弱窗口拖累，反而因为仍以 `ML` 为主导而继续被拉低。

### 结果三：最值得推进的修复方向不是“抛弃 ML”，而是只在 `trend_up_low_vol` 重配 ensemble
- `trend_up_low_vol_ml20_none30_v250`
  - 仅在 `trend_up_low_vol` 使用 `ml:0.20, none:0.30, v2:0.50`
  - 其余状态仍保留全局默认 `0.70 / 0.20 / 0.10`
  - 结果：
    - `full_excess_sharpe = 0.419`
    - `full_excess_total_return = +29.67%`
    - `weak_window_20250905_20260319_excess_sharpe = 0.684`
    - `weak_window_20250905_20260319_excess_total_return = +7.11%`
    - `trend_up_low_vol_weak_window_20250905_20260319_excess_sharpe = 0.558`
    - `trend_up_low_vol_weak_window_20250905_20260319_excess_total_return = +4.55%`
- 相对默认 `base_global`：
  - `full_excess_sharpe`: `0.265 -> 0.419`
  - `weak_window_20250905_20260319_excess_sharpe`: `-0.627 -> 0.684`
  - `trend_up_low_vol_weak_window_20250905_20260319_excess_sharpe`: `-0.667 -> 0.558`
- 次优但更偏全样本强势的候选是：
  - `trend_up_low_vol_ml50_none10_v240`
  - 结果：
    - `full_excess_sharpe = 0.420`
    - `full_excess_total_return = +38.78%`
    - `weak_window_20250905_20260319_excess_sharpe = 0.208`
    - `trend_up_low_vol_weak_window_20250905_20260319_excess_sharpe = 0.366`
- 这说明：
  - 当前最合理的修复方式不是全局改成 `v2_only`；
  - 而是保留 `ML` 作为全样本主干，同时只在 `trend_up_low_vol` 显著抬升 `v2`、下调 `ML`。

### 结果四：`ML` 和规则层并不重合，真正高度重合的是 `none` 与 `v2`
- 全样本 overlap 诊断：
  - `ml_vs_none`
    - `mean_pearson = 0.058`
    - `mean_spearman = 0.057`
    - `Top5 overlap = 1.02%`
  - `ml_vs_v2`
    - `mean_pearson = 0.073`
    - `mean_spearman = 0.094`
    - `Top5 overlap = 1.96%`
  - `none_vs_v2`
    - `mean_pearson = 0.960`
    - `mean_spearman = 0.953`
    - `Top5 overlap = 75.88%`
- 在 `trend_up_low_vol` 内：
  - `ml_vs_none`
    - `Top5 overlap = 1.30%`
  - `ml_vs_v2`
    - `Top5 overlap = 2.60%`
  - `none_vs_v2`
    - `Top5 overlap = 47.52%`
- 这说明：
  - `ML` 与规则层是明显不同的信号源；
  - `none` 与 `v2` 才是高度重叠的一对，`v2` 本质上更像 `none` 在 `trend_up_low_vol` 的局部修正版。

### 结果五：弱窗口里 `ML` 的“独立性”没有转化成优势，反而转化成拖累
- 在弱窗口 `2025-09-05 -> 2026-03-19`：
  - `ml_vs_none`
    - `Top5 overlap = 1.27%`
    - `ML Top5 - none Top5 = -15.70%`
  - `ml_vs_v2`
    - `Top5 overlap = 3.02%`
    - `ML Top5 - v2 Top5 = -10.73%`
- 在 `trend_up_low_vol` 且仍限定弱窗口时：
  - `ml_vs_none`
    - `Top5 overlap = 1.57%`
    - `ML Top5 - none Top5 = -17.57%`
  - `ml_vs_v2`
    - `Top5 overlap = 4.04%`
    - `ML Top5 - v2 Top5 = -11.96%`
- 这说明：
  - `ML` 的确提供了与规则层不同的排序；
  - 但在这段弱窗口里，这份“独立性”方向错了，带来的是负贡献而不是额外 alpha。

### 本轮结论
1. `ma50 baseline` 本身没有构成组合矛盾，真正的问题在于全局默认 ensemble 把 `ML` 放得太重。
2. 当前默认 `ml+none+v2 = 0.70 / 0.20 / 0.10` 已不应继续被视为“无需再碰”的稳态默认。
3. 目前最值得推进的正式修复方向是：
   - 仅在 `trend_up_low_vol` 下调 `ML`、上调 `v2`
   - 第一优先候选：`ml:0.20, none:0.30, v2:0.50`
   - 第二优先候选：`ml:0.50, none:0.10, v2:0.40`
4. 不建议把结论误读成“应全局移除 `ML`”：
   - 因为 `ml_only` 仍是全样本最强单体；
   - 真正合理的方向是“保留 ML 主线，但在 `trend_up_low_vol` 做状态专属降权”。

### 当前决策
1. 执行端默认暂不直接切换，但当前默认组合已进入“需要正式复验后再决定是否升级”的状态。
2. 下一轮若继续做正式研究，优先做更细的局部扫描，而不是再做全局盲扫：
   - 以 `trend_up_low_vol_ml20_none30_v250` 为中心
   - 用 `0.05` 步长在附近细扫
   - 重点扫描区间：
     - `ml 0.15 ~ 0.55`
     - `none 0.10 ~ 0.35`
     - `v2 0.30 ~ 0.60`
3. 后续文档与口头结论中，不再把 `latest_weak` 当作相对时间词使用，应写成：
   - `weak_window_20250905_20260319`

## 2026-03-26 `advanced_ml (ma50 baseline, lgbm)` `trend_up_low_vol` `0.05` 步长局部细扫补充复验
### 本轮动作
- 基于上一轮正式诊断结论，围绕 `trend_up_low_vol_ml20_none30_v250` 继续做状态专属 ensemble 局部细扫；
- 只保留 `state-only`，跳过已完成的 `ablation / overlap`，聚焦 `trend_up_low_vol` 的 `ml:none:v2` 局部权重结构；
- 正式输出目录：
  - `daily_research/output/advanced_ml_ensemble_focus_scan_20260326_local005`
- 本轮实际数据日期：
  - `latest_data_date = 2026-03-26`
- 本轮 `--auto-trim-history` 后实际使用的训练/评估历史窗口：
  - `2023-03-08 -> 2026-03-26`
- 本轮局部扫描约束：
  - `ml 0.15 ~ 0.55`
  - `none 0.10 ~ 0.35`
  - `v2 0.30 ~ 0.60`
  - `step = 0.05`
- 本轮弱窗口命名与指标列统一使用：
  - `weak_window_20250905_20260319`

### 结果一：上一轮中心候选有效，但局部最优点进一步收敛到 `ml 0.25 / none 0.20~0.25 / v2 0.50~0.55`
- `trend_up_low_vol_ml25_none25_v250`
  - `full_excess_sharpe = 0.829`
  - `full_excess_total_return = +67.85%`
  - `weak_window_20250905_20260319_excess_sharpe = 0.779`
  - `weak_window_20250905_20260319_excess_total_return = +9.26%`
  - `trend_up_low_vol_weak_window_20250905_20260319_excess_sharpe = 0.706`
  - `trend_up_low_vol_weak_window_20250905_20260319_excess_total_return = +6.65%`
- `trend_up_low_vol_ml25_none20_v255`
  - `full_excess_sharpe = 0.788`
  - `full_excess_total_return = +64.29%`
  - `weak_window_20250905_20260319_excess_sharpe = 0.914`
  - `weak_window_20250905_20260319_excess_total_return = +11.06%`
  - `trend_up_low_vol_weak_window_20250905_20260319_excess_sharpe = 0.881`
  - `trend_up_low_vol_weak_window_20250905_20260319_excess_total_return = +8.41%`
- `trend_up_low_vol_ml20_none30_v250`
  - `full_excess_sharpe = 0.689`
  - `full_excess_total_return = +52.25%`
  - `weak_window_20250905_20260319_excess_sharpe = 0.784`
  - `weak_window_20250905_20260319_excess_total_return = +8.63%`
  - `trend_up_low_vol_weak_window_20250905_20260319_excess_sharpe = 0.695`
  - `trend_up_low_vol_weak_window_20250905_20260319_excess_total_return = +6.04%`
- 这说明：
  - 上一轮选出来的 `trend_up_low_vol_ml20_none30_v250` 不是误报，它在更细扫描里仍处在前排；
  - 但局部前沿已经明显向 `ml 0.25`、`v2 0.50~0.55`、`none 0.20~0.25` 这一小块区域收敛。

### 结果二：与当前 `base_global` 相比，局部降 `ML`、抬 `v2` 仍然显著改善弱窗口
- `base_global`
  - `full_excess_sharpe = 0.732`
  - `full_excess_total_return = +73.66%`
  - `weak_window_20250905_20260319_excess_sharpe = -0.247`
  - `weak_window_20250905_20260319_excess_total_return = -4.77%`
  - `trend_up_low_vol_weak_window_20250905_20260319_excess_sharpe = -0.291`
  - `trend_up_low_vol_weak_window_20250905_20260319_excess_total_return = -4.30%`
- 相对 `base_global`，`trend_up_low_vol_ml25_none25_v250`：
  - `full_excess_sharpe`: `0.732 -> 0.829`
  - `weak_window_20250905_20260319_excess_sharpe`: `-0.247 -> 0.779`
  - `trend_up_low_vol_weak_window_20250905_20260319_excess_sharpe`: `-0.291 -> 0.706`
- 相对 `base_global`，`trend_up_low_vol_ml25_none20_v255`：
  - `full_excess_sharpe`: `0.732 -> 0.788`
  - `weak_window_20250905_20260319_excess_sharpe`: `-0.247 -> 0.914`
  - `trend_up_low_vol_weak_window_20250905_20260319_excess_sharpe`: `-0.291 -> 0.881`
- 这说明：
  - 默认组合在当前新数据口径下，全样本本身并不弱；
  - 但在 `weak_window_20250905_20260319` 这段里，它依然明显落后于 `trend_up_low_vol` 的状态专属降 `ML` 方案。

### 结果三：局部扫描已经把“平衡型候选”和“防守型候选”分开了
- 更平衡、可作为头号正式复验候选的是：
  - `trend_up_low_vol_ml25_none25_v250`
- 更偏弱窗口防守、但全样本仍不伤的候选是：
  - `trend_up_low_vol_ml25_none20_v255`
- 弱窗口 Sharpe 最高但全样本明显偏弱的极端防守候选是：
  - `trend_up_low_vol_ml15_none30_v255`
  - `full_excess_sharpe = 0.552`
  - `weak_window_20250905_20260319_excess_sharpe = 0.923`
- 这说明：
  - 当前已经不需要再做大范围盲扫；
  - 更合理的下一步，是把候选缩到 `ml25_none25_v250` 与 `ml25_none20_v255`，再和 `base_global` 做正式复验对照。

### 本轮结论
1. 细扫后，`trend_up_low_vol_ml20_none30_v250` 仍然成立，但已不再是局部最优点。
2. 当前最稳的平衡型候选更新为：`trend_up_low_vol_ml25_none25_v250`。
3. 当前最强的弱窗口防守型候选更新为：`trend_up_low_vol_ml25_none20_v255`。
4. 这轮结果继续支持同一方向：问题不在 `ma50 baseline`，而在默认全局 ensemble 在 `trend_up_low_vol` 里给了 `ML` 过高主导权。

### 当前决策
1. 执行端默认值仍不直接切换，先保留 `advanced_ml (ma50 baseline, lgbm) + liquid500 + next_open`。
2. 若继续正式复验，优先做三组严格对照：
   - `base_global`
   - `trend_up_low_vol_ml25_none25_v250`
   - `trend_up_low_vol_ml25_none20_v255`
3. 后续文档与口头结论继续统一使用：
   - `weak_window_20250905_20260319`

## 2026-03-27 `advanced_ml (ma50 baseline, lgbm)` 重训频率 / boosting 轮数正式敏感性实验
### 本轮目标
- 不再凭经验判断“`retrain_every_days` 和 `lgbm boosting rounds` 会不会影响结果”；
- 直接在当前执行主线口径下做正式实验，回答两件事：
  - 树模型重训频率是否影响结果；
  - `lgbm_n_estimators` 是否影响结果。

### 先修正一处本轮实验中暴露出来的时序错误
- 在第一次矩阵实验 `advanced_ml_retrain_tree_impact_20260326_formal_r1` 里，出现了不合理的超高 Sharpe；
- 回查后确认，问题不在参数本身，而在 `rolling ML` 训练窗口的标签边界：
  - 训练样本末端没有对 `target_horizon` 做足够的 label-safe 截断；
  - 会把对预测块而言“未来才知道”的标签提前喂进滚动训练。
- 已修正为：
  - `train_point_in_time_model` 与 `rolling_ml_scores` 都按 `target_horizon` 与 `execution_mode=next_open` 做 label-safe 截断；
  - 共享 `ml score cache` 也同步抬了版本，避免误命中旧缓存。
- 因此：
  - `advanced_ml_retrain_tree_impact_20260326_formal_r1` 作废；
  - 本轮正式结论只以 `advanced_ml_retrain_tree_impact_20260326_formal_r2` 为准。

### 本轮动作
- 新增正式实验脚本：
  - `daily_research/baseline/scan_advanced_ml_retrain_tree_impact.py`
- 这轮固定只比较三组候选，不再做全局盲扫：
  - `base_global`
  - `trend_up_low_vol_ml25_none25_v250`
  - `trend_up_low_vol_ml25_none20_v255`
- 正式输出目录：
  - `daily_research/output/advanced_ml_retrain_tree_impact_20260326_formal_r2`
- 本轮实际数据日期：
  - `latest_data_date = 2026-03-26`
- 本轮 `--auto-trim-history` 后实际训练/评估历史窗口：
  - `2023-03-08 -> 2026-03-26`
- 固定比较网格：
  - `retrain_every_days = 5, 10, 21, 42`
  - `lgbm_n_estimators = 130, 260, 520`
- 本轮弱窗口命名统一使用：
  - `weak_window_20250905_20260319`

### 结果一：结论已经实验确认，两类参数都会影响结果
- `base_global`
  - `full_excess_sharpe` 全范围：`-0.351 -> 0.248`
  - `weak_window_20250905_20260319_excess_sharpe` 全范围：`-1.922 -> -0.486`
- `trend_up_low_vol_ml25_none20_v255`
  - `full_excess_sharpe` 全范围：`-0.176 -> 0.282`
  - `weak_window_20250905_20260319_excess_sharpe` 全范围：`-0.581 -> 0.346`
- `trend_up_low_vol_ml25_none25_v250`
  - `full_excess_sharpe` 全范围：`0.006 -> 0.311`
  - `weak_window_20250905_20260319_excess_sharpe` 全范围：`-0.308 -> 0.425`
- 这说明：
  - `retrain_every_days` 不是无关参数；
  - `lgbm_n_estimators` 也不是无关参数；
  - 两者都会实质改变结论，不能再把当前 `21 / 260` 当成默认不动的“天然合理值”。

### 结果二：`base_global` 会被参数调整拉动，但仍然修不好弱窗口
- 默认 `base_global @ 21 / 260`
  - `full_excess_sharpe = -0.008`
  - `weak_window_20250905_20260319_excess_sharpe = -0.939`
- `base_global` 全样本最强点在 `5 / 260`
  - `full_excess_sharpe = 0.248`
  - `weak_window_20250905_20260319_excess_sharpe = -0.486`
- `base_global` 的弱窗口最佳点也仍然是 `5 / 260`
  - `weak_window_20250905_20260319_excess_sharpe = -0.486`
  - `trend_up_low_vol_weak_window_20250905_20260319_excess_sharpe = -0.612`
- 这说明：
  - 更频繁重训确实能改善 `base_global`；
  - 但 `base_global` 即使调到本轮最佳参数，弱窗口仍然是负 Sharpe，不能靠“只调训练次数”完成修复。

### 结果三：状态专属候选会被参数显著放大，而且已经出现弱窗口转正
- 默认 `trend_up_low_vol_ml25_none25_v250 @ 21 / 260`
  - `full_excess_sharpe = 0.158`
  - `weak_window_20250905_20260319_excess_sharpe = -0.131`
- 它的全样本最强点在 `5 / 130`
  - `full_excess_sharpe = 0.311`
  - `weak_window_20250905_20260319_excess_sharpe = 0.005`
- 它的弱窗口最强点在 `5 / 520`
  - `full_excess_sharpe = 0.222`
  - `weak_window_20250905_20260319_excess_sharpe = 0.425`
  - `trend_up_low_vol_weak_window_20250905_20260319_excess_sharpe = 0.253`
- 默认 `trend_up_low_vol_ml25_none20_v255 @ 21 / 260`
  - `full_excess_sharpe = -0.035`
  - `weak_window_20250905_20260319_excess_sharpe = -0.581`
- 它的全样本最强点在 `5 / 260`
  - `full_excess_sharpe = 0.282`
  - `weak_window_20250905_20260319_excess_sharpe = 0.312`
- 它的弱窗口最强点在 `21 / 520`
  - `full_excess_sharpe = 0.173`
  - `weak_window_20250905_20260319_excess_sharpe = 0.346`
  - `trend_up_low_vol_weak_window_20250905_20260319_excess_sharpe = 0.150`
- 这说明：
  - 两个状态专属候选都不是“参数不敏感”的假稳态；
  - 但它们都能在更合适的参数组合下，把弱窗口从负值推到正值；
  - 其中 `ml25_none25_v250` 当前是更稳的主候选。

### 结果四：重训频率通常对全样本更敏感，boosting 轮数对弱窗口常常同样重要甚至更重要
- 用组均值看参数影响范围：
  - `trend_up_low_vol_ml25_none25_v250`
    - `full_excess_sharpe`：
      - `retrain` 组均值范围 `0.154`
      - `n_estimators` 组均值范围 `0.057`
    - `weak_window_20250905_20260319_excess_sharpe`：
      - `retrain` 组均值范围 `0.234`
      - `n_estimators` 组均值范围 `0.224`
  - `trend_up_low_vol_ml25_none20_v255`
    - `full_excess_sharpe`：
      - `retrain` 组均值范围 `0.198`
      - `n_estimators` 组均值范围 `0.106`
    - `weak_window_20250905_20260319_excess_sharpe`：
      - `retrain` 组均值范围 `0.318`
      - `n_estimators` 组均值范围 `0.362`
- 这说明：
  - 对全样本表现，`retrain_every_days` 往往影响更大；
  - 对弱窗口表现，`lgbm_n_estimators` 的影响并不比重训频率小，有时还更大；
  - 所以以后不能只扫一个维度，至少要把这两个维度联动看。

### 本轮结论
1. “树模型重训频率 + boosting 轮数是否有影响”这个问题，现在已经有正式实验答案：有，而且是实质影响，不是边角扰动。
2. 当前默认 `21 / 260` 并不是这三组候选里的稳定优值。
3. `base_global` 可以通过参数优化改善，但仍然不能解决 `weak_window_20250905_20260319` 的核心弱点。
4. 当前更值得继续推进的是：
   - `trend_up_low_vol_ml25_none25_v250 @ 5 / 520`
   - `trend_up_low_vol_ml25_none20_v255 @ 5 / 260`
   - 这两者都比默认 `21 / 260` 更值得正式复验。
5. 本轮也顺手确认了一件更底层的事：
   - 以后凡是基于 rolling ML 的正式研究，必须沿用这次修正后的 label-safe 边界；
   - 不能再引用修正前那版滚动结果。

### 当前决策
1. 执行端默认值仍不直接切换，先保留 `advanced_ml (ma50 baseline, lgbm) + liquid500 + next_open`。
2. 若继续正式复验，优先仍是这三组严格对照，但不再默认锁死 `21 / 260`：
   - `base_global`
   - `trend_up_low_vol_ml25_none25_v250`
   - `trend_up_low_vol_ml25_none20_v255`
3. 下一轮正式复验时，至少应把以下参数组合纳入首轮：
   - `5 / 260`
   - `5 / 520`
   - `21 / 520`
4. 后续文档与口头结论继续统一使用：
   - `weak_window_20250905_20260319`

## 2026-03-28 `advanced_ml (ma50 baseline, lgbm)` 训练窗口 / 重训频率 / boosting 轮数正式矩阵
### 本轮目标
- 把 `ml_train_window_days` 正式纳入当前执行主线的受控变量。
- 不再只看 `retrain_every_days` 与 `lgbm_n_estimators`，而是一起回答：
  - `ml_train_window_days` 是否实质影响结果；
  - 三个当前候选在 `train_window / retrain / n_estimators` 联动下的最优点分别在哪里。

### 先修正本轮第一版矩阵的历史窗口边界
- 2026-03-27 先跑出的 `advanced_ml_retrain_tree_impact_20260327_trainwindow_formal_r1` 不作为正式结论引用。
- 原因不是候选本身失效，而是脚本刚接入 `--train-window-grid` 后，`--auto-trim-history` 仍按默认 `ml_train_window_days=504` 裁历史；
- 但本轮网格已经扩到 `train_window_days=756`，导致 `756` 这组历史不够长，训练日志几乎为空，结果更接近规则层回退，不适合作为正式对照。
- 已修正为：
  - `auto-trim-history` 按 `train-window-grid` 的最大窗口裁历史；
  - 本轮正式口径只认 `advanced_ml_retrain_tree_impact_20260327_trainwindow_formal_r2`。

### 本轮动作
- 脚本：
  - `daily_research/baseline/scan_advanced_ml_retrain_tree_impact.py`
- 新增能力：
  - `--train-window-grid`
  - 摘要 / 默认行 / 最优行 / run_config 全部带上 `ml_train_window_days`
  - `auto-trim-history` 现在按 `train-window-grid` 最大值取历史窗口
- 正式输出目录：
  - `daily_research/output/advanced_ml_retrain_tree_impact_20260327_trainwindow_formal_r2`
- 实际最新数据日期：
  - `latest_data_date = 2026-03-27`
- 本轮 `--auto-trim-history` 后的实际训练/评估历史窗口：
  - `20220322 -> 20260327`
- 固定候选仍只看三组严格对照：
  - `base_global`
  - `trend_up_low_vol_ml25_none25_v250`
  - `trend_up_low_vol_ml25_none20_v255`
- 本轮正式参数网格：
  - `ml_train_window_days = 378, 504, 756`
  - `retrain_every_days = 5, 21`
  - `lgbm_n_estimators = 260, 520`
- 后续命名继续统一：
  - `weak_window_20250905_20260319`

### 结果一：`ml_train_window_days` 确实会影响结果，而且不是越长越好
- `base_global`
  - 默认 `504 / 21 / 260`: `full_excess_sharpe = 0.410`
  - 默认 `504 / 21 / 260`: `weak_window_20250905_20260319_excess_sharpe = -0.961`
  - 最强全样本点在 `504 / 5 / 260`: `0.837 / -0.788`
  - 最强弱窗口点在 `378 / 21 / 520`: `0.776 / -0.383`
- `trend_up_low_vol_ml25_none20_v255`
  - 默认 `504 / 21 / 260`: `0.515 / 0.368`
  - 最强全样本点在 `504 / 5 / 260`: `0.793 / 0.204`
  - 最强弱窗口点在 `504 / 21 / 520`: `0.699 / 1.094`
- `trend_up_low_vol_ml25_none25_v250`
  - 默认 `504 / 21 / 260`: `0.549 / 0.692`
  - 最强全样本点在 `504 / 5 / 260`: `0.745 / -0.070`
  - 最强弱窗口点在 `504 / 21 / 520`: `0.663 / 1.154`
- 这说明：
  - `ml_train_window_days` 不是无关变量；
  - 但当前执行主线下，并没有出现“窗口拉到 `756` 就自然更稳”的结论；
  - 在这轮正式矩阵里，真正占优的主轴仍然落在 `504`。

### 结果二：`504` 是当前两组状态专属候选的主窗口，`756` 不是
- 按弱窗口最优点看：
  - `trend_up_low_vol_ml25_none20_v255`
    - `378` 最优：`0.427 / 0.314`
    - `504` 最优：`0.699 / 1.094`
    - `756` 最优：`0.329 / 0.827`
  - `trend_up_low_vol_ml25_none25_v250`
    - `378` 最优：`0.485 / 0.152`
    - `504` 最优：`0.663 / 1.154`
    - `756` 最优：`0.355 / 0.801`
- 这说明：
  - `756` 不是没用，它确实能把弱窗口维持在正值；
  - 但在这轮正式口径里，`504` 明显更强，尤其配合 `21 / 520` 时，已经同时兼顾了全样本与弱窗口。

### 结果三：当前最值得推进的正式候选进一步收敛
- `base_global`
  - 即使纳入 `train_window_days`，弱窗口最优也仍为负值；
  - 说明它可以被参数改善，但依旧不是能够修复当前弱窗口问题的主候选。
- `trend_up_low_vol_ml25_none20_v255`
  - 当前最强弱窗口点更新为：
    - `504 / 21 / 520`
    - `full_excess_sharpe = 0.699`
    - `weak_window_20250905_20260319_excess_sharpe = 1.094`
    - `trend_up_low_vol_weak_window_20250905_20260319_excess_sharpe = 1.076`
- `trend_up_low_vol_ml25_none25_v250`
  - 当前最强弱窗口点更新为：
    - `504 / 21 / 520`
    - `full_excess_sharpe = 0.663`
    - `weak_window_20250905_20260319_excess_sharpe = 1.154`
    - `trend_up_low_vol_weak_window_20250905_20260319_excess_sharpe = 1.139`
- 这说明：
  - 这两组状态专属候选仍然是当前最值得推进的正式复验对象；
  - 而且在引入 `ml_train_window_days` 之后，它们并没有被推翻，反而更清楚地收敛到 `504 / 21 / 520`。

### 本轮结论
1. `ml_train_window_days` 会实质影响结果，这个问题现在已经有正式实验答案。
2. 在本轮正式矩阵里，`504` 明显强于 `378` 与 `756`，当前没有证据支持把训练窗口继续机械拉长到 `756`。
3. `base_global` 仍不具备作为弱窗口修复主方案的资格。
4. 当前最值得推进正式复验的两组候选，进一步收敛为：
   - `trend_up_low_vol_ml25_none25_v250 @ 504 / 21 / 520`
   - `trend_up_low_vol_ml25_none20_v255 @ 504 / 21 / 520`
5. `504 / 5 / 260` 依然有价值，但更像“全样本进攻型对照点”，不如 `504 / 21 / 520` 兼顾弱窗口稳定性。

### 当前决策
1. 执行端默认值仍不直接切换，继续保留 `advanced_ml (ma50 baseline, lgbm) + liquid500 + next_open`。
2. 若继续正式复验，三组严格对照仍保持不变：
   - `base_global`
   - `trend_up_low_vol_ml25_none25_v250`
   - `trend_up_low_vol_ml25_none20_v255`
3. 下一轮正式 pair revalidation 时，参数优先级更新为：
   - `504 / 21 / 520`
   - `504 / 5 / 260`
   - `378 / 21 / 520`
4. 后续文档与口头结论继续统一使用：
   - `weak_window_20250905_20260319`

## 2026-03-28 `advanced_ml (ma50 baseline, lgbm)` 长窗口 pair revalidation
### 本轮目标
- 对上一轮已经收敛出的三组参数优先级做长窗口正式复验，不再做新的参数盲扫。
- 本轮只验证三组显式参数：
  - `504 / 21 / 520`
  - `504 / 5 / 260`
  - `378 / 21 / 520`

### 先说明本轮第一版 `r1`
- 2026-03-28 先跑出的 `advanced_ml_pair_revalidation_20260328_formal_r1` 不作为长窗口正式结论引用。
- 原因是当时仍带了 `--auto-trim-history`，实际历史窗口被自动裁回：
  - `20230309 -> 20260327`
- 它可以作为“这三组参数在当前短样本上的快速对照”，但不是这轮要的长窗口 pair revalidation。
- 本轮正式口径只认：
  - `advanced_ml_pair_revalidation_20260328_formal_r2`

### 本轮动作
- 脚本继续使用：
  - `daily_research/baseline/scan_advanced_ml_retrain_tree_impact.py`
- 为了精确复验三组指定参数，脚本新增：
  - `--explicit-configs`
- 正式输出目录：
  - `daily_research/output/advanced_ml_pair_revalidation_20260328_formal_r2`
- 最新数据日期：
  - `latest_data_date = 2026-03-27`
- 本轮正式历史窗口：
  - `20190101 -> 20260327`
- 固定候选仍只看三组严格对照：
  - `base_global`
  - `trend_up_low_vol_ml25_none25_v250`
  - `trend_up_low_vol_ml25_none20_v255`
- 命名继续统一：
  - `weak_window_20250905_20260319`

### 结果一：长窗口复验后，只有 `504 / 21 / 520` 真正站住
- `base_global`
  - `378 / 21 / 520`: `full_excess_sharpe = -0.111`, `weak_window_20250905_20260319_excess_sharpe = -0.581`
  - `504 / 5 / 260`: `0.375 / -0.544`
  - `504 / 21 / 520`: `0.016 / -0.656`
- `trend_up_low_vol_ml25_none20_v255`
  - `378 / 21 / 520`: `0.012 / -0.876`
  - `504 / 5 / 260`: `0.550 / -0.575`
  - `504 / 21 / 520`: `0.860 / 0.819`
- `trend_up_low_vol_ml25_none25_v250`
  - `378 / 21 / 520`: `0.100 / -0.798`
  - `504 / 5 / 260`: `0.492 / -0.287`
  - `504 / 21 / 520`: `0.786 / 0.992`
- 这说明：
  - 在长窗口 pair revalidation 里，`504 / 21 / 520` 是唯一能让两组状态专属候选同时保持“全样本强 + 弱窗口为正”的参数；
  - `504 / 5 / 260` 和 `378 / 21 / 520` 在长窗口下都没有守住弱窗口。

### 结果二：两组状态专属候选都通过了长窗口复验，但侧重点不同
- `trend_up_low_vol_ml25_none20_v255 @ 504 / 21 / 520`
  - `full_excess_sharpe = 0.860`
  - `weak_window_20250905_20260319_excess_sharpe = 0.819`
  - `trend_up_low_vol_weak_window_20250905_20260319_excess_sharpe = 0.751`
- `trend_up_low_vol_ml25_none25_v250 @ 504 / 21 / 520`
  - `full_excess_sharpe = 0.786`
  - `weak_window_20250905_20260319_excess_sharpe = 0.992`
  - `trend_up_low_vol_weak_window_20250905_20260319_excess_sharpe = 0.968`
- 这说明：
  - `ml25_none20_v255` 更偏全样本进攻；
  - `ml25_none25_v250` 更偏弱窗口和 `trend_up_low_vol` 防守；
  - 两者都已经明显优于 `base_global`。

### 本轮结论
1. 长窗口 pair revalidation 已经把参数层面收敛到一个很清楚的结论：
   - `504 / 21 / 520`
2. `504 / 5 / 260` 没有通过长窗口复验，不再适合作为优先晋级参数。
3. `378 / 21 / 520` 也没有通过长窗口复验，不再适合作为优先晋级参数。
4. 当前真正通过长窗口正式复验、值得进入执行端升级 shortlist 的，只剩两组：
   - `trend_up_low_vol_ml25_none25_v250 @ 504 / 21 / 520`
   - `trend_up_low_vol_ml25_none20_v255 @ 504 / 21 / 520`
5. 若按弱窗口稳健性排序，当前更强的是：
   - `trend_up_low_vol_ml25_none25_v250 @ 504 / 21 / 520`
6. 若按全样本收益排序，当前更强的是：
   - `trend_up_low_vol_ml25_none20_v255 @ 504 / 21 / 520`

### 当前决策
1. 执行端默认值暂不自动切换，但“是否升级”的判断门槛已经满足。
2. 后续若继续推进，不再需要回头做这三组参数的重复复验。
3. 下一步应直接聚焦两组最终候选的 head-to-head：
   - `trend_up_low_vol_ml25_none25_v250 @ 504 / 21 / 520`
   - `trend_up_low_vol_ml25_none20_v255 @ 504 / 21 / 520`
4. 后续文档与口头结论继续统一使用：
   - `weak_window_20250905_20260319`

## 2026-03-28 `advanced_ml (ma50 baseline, lgbm)` shortlist head-to-head
### 本轮目标
- 把已经通过长窗口正式复验的两组最终候选，收口成一次可复用、可回滚、可解释的正式 head-to-head。
- 不再扩新候选，不再重跑已经淘汰的参数组，只比较：
  - `trend_up_low_vol_ml25_none25_v250 @ 504 / 21 / 520`
  - `trend_up_low_vol_ml25_none20_v255 @ 504 / 21 / 520`

### 本轮动作
- 新增汇总脚本：
  - `daily_research/baseline/render_advanced_ml_shortlist_head_to_head.py`
- 直接读取两份正式输出：
  - `daily_research/output/advanced_ml_retrain_tree_impact_20260327_trainwindow_formal_r2`
  - `daily_research/output/advanced_ml_pair_revalidation_20260328_formal_r2`
- 固定精确配置：
  - `504 / 21 / 520`
- 正式 head-to-head 输出目录：
  - `daily_research/output/advanced_ml_shortlist_head_to_head_20260328_formal_r1`
- 本轮继续统一使用：
  - `weak_window_20250905_20260319`

### 结果一：两组最终候选的优势分工被正式固定
- `trend_up_low_vol_ml25_none20_v255`
  - 在两份 formal 输出里，`full_excess_sharpe` 都更强：
    - `0.699 > 0.663`
    - `0.860 > 0.786`
- `trend_up_low_vol_ml25_none25_v250`
  - 在两份 formal 输出里，`weak_window_20250905_20260319_excess_sharpe` 都更强：
    - `1.154 > 1.094`
    - `0.992 > 0.819`
  - 在两份 formal 输出里，`trend_up_low_vol_weak_window_20250905_20260319_excess_sharpe` 也都更强：
    - `1.139 > 1.076`
    - `0.968 > 0.751`
- 这说明：
  - `ml25_none20_v255` 的优势继续稳定落在全样本进攻；
  - `ml25_none25_v250` 的优势继续稳定落在弱窗口与 `trend_up_low_vol` 防守。

### 结果二：辅助指标没有把结论收口成单一赢家
- `recent_full_excess_sharpe`
  - 在短历史 formal 矩阵里是 `v250` 更强；
  - 在长窗口 pair revalidation 里是 `v255` 更强。
- `full_excess_max_drawdown` 与 `recent_full_excess_max_drawdown`
  - 也没有形成跨两份 formal 输出都偏向同一候选的单边优势。
- `full_avg_turnover`
  - `v255` 略优，但差距很小。
- 这说明：
  - 当前并不存在一个在“全样本收益、弱窗口稳健、回撤、换手”上同时形成单边优势的候选；
  - 这轮 head-to-head 产出的不是“升级赢家”，而是“分工清楚但仍 split 的 verdict”。

### 本轮结论
1. 当前 formal head-to-head 已经完成，但没有形成单一升级赢家。
2. 若按全样本进攻排序，当前更强的是：
   - `trend_up_low_vol_ml25_none20_v255 @ 504 / 21 / 520`
3. 若按弱窗口与 `trend_up_low_vol` 防守排序，当前更强的是：
   - `trend_up_low_vol_ml25_none25_v250 @ 504 / 21 / 520`
4. 按当前项目停止规则执行，结论应当是：
   - 维持现默认值，不做口头升级。

### 当前决策
1. 执行端默认值继续保持：
   - `advanced_ml (ma50 baseline, lgbm) + liquid500 + next_open`
2. 后续若继续推进升级，不再重开更大候选分支。
3. 下一步若仍要推动升级，先做以下二选一：
   - 明确写出“全样本进攻 vs 弱窗口防守”谁是当前升级优先级；
   - 或补一项同口径的 formal comparator，再让两组最终候选分出单一赢家。
4. 后续文档与口头结论继续统一使用：
   - `weak_window_20250905_20260319`

## 2026-03-28 `advanced_ml (ma50 baseline, lgbm)` 攻守控制器首轮正式扫描
### 本轮目标
- 不再逼 `v250` 与 `v255` 选出静态唯一赢家。
- 直接验证一个最小可执行的动态控制器：在 `trend_up_low_vol` 内部，根据市场趋势强度与波动水平，在两组最终候选之间切换。

### 本轮动作
- 新增脚本：
  - `daily_research/baseline/scan_advanced_ml_attack_defense_controller.py`
- 正式输出目录：
  - `daily_research/output/advanced_ml_attack_defense_controller_20260328_formal_r1`
- 正式口径继续固定为：
  - `liquid500`
  - `next_open`
  - `20190101 -> 20260327`
  - `504 / 21 / 520`
- 静态对照仍只看：
  - `trend_up_low_vol_ml25_none25_v250`
  - `trend_up_low_vol_ml25_none20_v255`
- 动态控制器首轮只做最小扫描：
  - 在 `trend_up_low_vol` 内，若 `trend_gap >= 阈值` 且 `annual_vol <= 阈值`，则切到 `v255`
  - 否则保持 `v250`
- 首轮阈值网格：
  - `trend_gap = 0.010, 0.024, 0.044, 0.065`
  - `annual_vol = 0.140, 0.170, 0.200, 0.320`

### 结果一：当前没有动态控制器能同时压过两组静态 shortlist
- 静态 `v250`
  - `full_excess_sharpe = 0.786`
  - `weak_window_20250905_20260319_excess_sharpe = 0.992`
  - `trend_up_low_vol_weak_window_20250905_20260319_excess_sharpe = 0.968`
- 静态 `v255`
  - `full_excess_sharpe = 0.860`
  - `weak_window_20250905_20260319_excess_sharpe = 0.819`
  - `trend_up_low_vol_weak_window_20250905_20260319_excess_sharpe = 0.751`
- 本轮 16 个动态候选里：
  - `dominating_dynamic_candidate_count = 0`
- 这说明：
  - 首轮简单控制器还不能把“进攻”和“防守”同时收口成一个可直接升级的默认方案。

### 结果二：简单控制器确实有信号，但还不够形成默认值升级
- 进攻最强的动态候选：
  - `gap>=0.010, vol<=0.320`
  - `full_excess_sharpe = 0.817`
  - `weak_window_excess_sharpe = 0.745`
  - `focus_weak_excess_sharpe = 0.657`
- 防守最强、也是 balance score 最强的动态候选：
  - `gap>=0.024, vol<=0.140`
  - `full_excess_sharpe = 0.758`
  - `weak_window_excess_sharpe = 1.095`
  - `focus_weak_excess_sharpe = 1.108`
- 当前最像“折中型”的候选之一：
  - `gap>=0.024, vol<=0.200`
  - `full_excess_sharpe = 0.806`
  - `weak_window_excess_sharpe = 0.987`
  - `focus_weak_excess_sharpe = 0.966`
- 这说明：
  - 简单控制器已经能明显改变攻守平衡；
  - 但它还做不到既保住 `v255` 的全样本上沿，又同时稳定压过 `v250` 的弱窗口上沿。

### 本轮结论
1. “做动态攻守控制器”这条方向本身是对的，不需要回退到静态二选一。
2. 但首轮最小规则版控制器还不够作为默认执行升级方案。
3. 当前最重要的真实信息不是“动态无效”，而是：
   - 简单 `trend_gap + annual_vol` 双阈值规则不足以完成这次升级。
4. 后续若继续推进，重点应转向：
   - 补强 `trend_up_low_vol` 内部的状态识别与切换信号；
   - 而不是继续在同一层重复更密的纯阈值扫网格。

### 当前决策
1. 执行端默认值继续保持：
   - `advanced_ml (ma50 baseline, lgbm) + liquid500 + next_open`
2. 动态控制器方向继续保留为优先研究线。
3. 下一步不再回头做静态 shortlist 的重复争论。
4. 下一步若继续推进，优先做：
   - 更强的 `trend_up_low_vol` 内部攻守识别信号
   - 再用同一正式口径重跑动态控制器正式比较

## 2026-03-28 `advanced_ml (ma50 baseline, lgbm)` 攻守分型诊断
### 本轮目标
- 不再停留在“动态方案有没有赢家”这一层。
- 直接拆清 `trend_up_low_vol` 内，究竟哪些信号更像正式攻守切换入口，哪些只是解释变量。

### 本轮动作
- 新增脚本：
  - `daily_research/baseline/diagnose_advanced_ml_attack_defense.py`
- 正式输出目录：
  - `daily_research/output/advanced_ml_attack_defense_diagnosis_20260328_formal_r1`
- 诊断对象继续固定为：
  - 静态 `trend_up_low_vol_ml25_none20_v255`
  - 静态 `trend_up_low_vol_ml25_none25_v250`
- 诊断口径继续固定为：
  - `liquid500`
  - `next_open`
  - `20190101 -> 20260327`
  - `504 / 21 / 520`

### 结果一：日级攻守差分真实存在，但强度还不够直接形成单变量 gate
- `focus_state_days = 482`
- `offense_minus_defense_mean_excess_return` 接近零轴，但 `offense_win_rate = 0.523`
- 这说明：
  - `v255` 与 `v250` 的日级优势差异不是不存在；
  - 但它不是那种用单一简单特征就能一下子完全分开的强信号。

### 结果二：`focus_streak` 更像解释变量，不像下一轮 formal controller 的主入口
- `focus_streak>=18 / 27 / 10` 这类单规则虽然在全样本上有轻微正向 mean diff；
- 但它们在弱窗口里的 `weak_mean_diff` 并不稳定，普遍没有形成比 `ret10` 更清晰的正式升级方向。
- 这说明：
  - `focus_streak` 可以保留为解释 `trend_up_low_vol` 内部节奏的诊断特征；
  - 但当前不值得直接升格为第二轮正式 gating 入口。

### 结果三：更值得 formal 化的是 `benchmark_ret_10d`
- 诊断里更像“下一轮正式入口”的组合集中在：
  - `trend_gap >= 0.024192`
  - `annual_vol <= 0.176128`
  - 再叠加 `benchmark_ret_10d`
- 这说明：
  - 与其继续在首轮 `trend_gap + annual_vol` 上加密网格；
  - 不如直接把 `benchmark_ret_10d` 作为第二代动态控制器的新增门槛进入正式复验。

### 本轮结论
1. 当前攻守控制器方向继续成立，不回退到静态二选一。
2. 下一轮最值得 formal 化的新轴不是 `focus_streak`，而是 `benchmark_ret_10d`。
3. 因此下一步直接进入：
   - 带 `ret10` 门槛的第二轮正式动态控制器扫描。

### 当前决策
1. 保留 `focus_streak` 为诊断变量。
2. 把 `benchmark_ret_10d` 升为第二轮正式控制器扫描的新增门槛。

## 2026-03-28 `advanced_ml (ma50 baseline, lgbm)` 攻守控制器第二轮正式扫描（加 `ret10`）
### 本轮目标
- 验证在首轮 `trend_gap + annual_vol` 基础上，再加入 `benchmark_ret_10d`，能否把动态控制器正式推到可升级默认值的水平。

### 本轮动作
- 继续使用脚本：
  - `daily_research/baseline/scan_advanced_ml_attack_defense_controller.py`
- 脚本新增：
  - `--offense-benchmark-ret10-grid`
- 正式输出目录：
  - `daily_research/output/advanced_ml_attack_defense_controller_20260328_formal_r2_ret10`
- 第二轮正式网格收敛为：
  - `trend_gap = 0.015576, 0.024192`
  - `annual_vol = 0.108317, 0.176128`
  - `benchmark_ret_10d = -0.007665, 0.003449, 0.014717`
- 静态对照继续保持：
  - `trend_up_low_vol_ml25_none25_v250`
  - `trend_up_low_vol_ml25_none20_v255`

### 结果一：`ret10` 确实让动态方案更接近正式可用
- `dominating_dynamic_candidate_count = 0`
- 但第二轮最强折中候选已经变成：
  - `gap>=0.024192, vol<=0.176128, ret10>=0.014717`
  - `full_excess_sharpe = 0.807`
  - `weak_window_20250905_20260319_excess_sharpe = 1.101`
  - `trend_up_low_vol_weak_window_20250905_20260319_excess_sharpe = 1.114`
- 这说明：
  - `ret10` 不是噪音门槛；
  - 它确实把动态方案的弱窗口与 focus-weak 防守能力又往上推了一层。

### 结果二：当前更偏进攻的第二轮候选也已经比首轮更完整
- 本轮 full 端最强动态候选是：
  - `gap>=0.024192, vol<=0.176128, ret10>=0.003449`
  - `full_excess_sharpe = 0.818`
  - `weak_window_excess_sharpe = 1.088`
  - `focus_weak_excess_sharpe = 1.097`
- 这说明：
  - 第二轮 `ret10` 控制器已经不再只是“纯防守补丁”；
  - 它开始形成“full 端不算差，弱窗口端明显更强”的正式动态候选形态。

### 结果三：但它还没有真正跨过升级门槛
- 静态 `v255` 仍然保持：
  - `full_excess_sharpe = 0.860`
- 静态 `v250` 仍然保持：
  - `weak_window_excess_sharpe = 0.992`
  - `focus_weak_excess_sharpe = 0.968`
- 当前第二轮最强动态候选虽然已经在弱窗口侧明显超过 `v250`；
- 但在 full 端仍没超过静态 `v255`。
- 这说明：
  - 当前问题已经从“动态方向有没有信号”收敛成：
  - 如何在保住 `ret10` 带来的弱窗口增益前提下，再把 full 端补回去。

### 本轮结论
1. 第二轮正式扫描进一步确认：
   - `benchmark_ret_10d` 是值得保留的第二代动态控制轴。
2. 当前最强动态折中候选是：
   - `gap>=0.024192, vol<=0.176128, ret10>=0.014717`
3. 当前更偏 full 端的动态候选是：
   - `gap>=0.024192, vol<=0.176128, ret10>=0.003449`
4. 但当前仍没有一个动态控制器能同时压过静态 `v255` 的 full 上沿与静态 `v250` 的弱窗口防守。
5. 因此执行端默认值继续保持不切换。

### 当前决策
1. 执行端默认值继续保持：
   - `advanced_ml (ma50 baseline, lgbm) + liquid500 + next_open`
2. 动态控制器研究线继续保留，而且优先级高于回头争论静态二选一。
3. 下一步不再把 `focus_streak` 升格为正式 gating 入口。
4. 下一步若继续推进，优先做：
   - 围绕 `gap>=0.024192, vol<=0.176128, ret10>=0.014717 / 0.003449` 继续补 full 端收益；
   - 在保住这层 `ret10` 弱窗口增益的前提下，再做同口径 formal comparator。

> 注：以上结论基于旧 `market_features` 口径，已被下方“当前代码口径重跑 + 执行默认值升级”条目覆盖；后续默认以该新条目为准。

## 2026-03-28 `advanced_ml (ma50 baseline, lgbm)` 当前代码口径重跑 + 执行默认值升级
### 本轮目标
- 解释为什么旧 formal `r1 / r2` 与当前脚本结果发生冲突；
- 在当前代码、当前特征空间和当前 rolling ML score 口径下，重新决定 live execution default。

### 本轮动作
- 正式输出目录：
  - `daily_research/output/advanced_ml_attack_defense_controller_20260328_formal_r3_weightgrid_focus`
  - `daily_research/output/advanced_ml_attack_defense_controller_20260328_formal_r3_baseprobe`
- 对比了旧 formal 产物与当前 prepared / ml_score cache：
  - 旧 prepared cache：`daily_research/cache/advanced_ml/prepared/2ec8e125112e2645a3aa.pkl`
  - 当前 prepared cache：`daily_research/cache/advanced_ml/prepared/e237c99bfbfa9a984e00.pkl`
  - 旧 ml_scores cache：`daily_research/cache/advanced_ml/ml_scores/5b68d4a05901bf6f1229.pkl`
  - 当前 ml_scores cache：`daily_research/cache/advanced_ml/ml_scores/bb3aae2219e0720b81ca.pkl`
- 同时把执行端 wrapper 的共享默认参数切到：
  - `trend_up_low_vol=ml:0.25,none:0.25,v2:0.50`
- 并重训 live artifact：
  - `daily_research/execution/models/latest_ml_model.joblib`
  - `daily_research/execution/models/latest_ml_model.json`

### 结果一：旧 split verdict 失效的原因，不在四象限，也不在规则层，而在 market-feature 扩容后的 ML 分数漂移
- 旧 prepared / 当前 prepared 之间：
  - `regime_state` 行数一致；
  - `trend_up_low_vol` 天数一致；
  - `score_none / score_v2 / filter_mask` 一致；
  - `feature_frames` 一致；
  - 但 `market_features` 已从 `7` 个扩到 `24` 个。
- 当前新增的 market features 里，包括：
  - `benchmark_vol_gap`
  - `benchmark_vol_ratio`
  - `trend_bucket / vol_bucket` 对应的一组布尔 market signals
- 这直接导致 rolling ML scores 已经实质变化：
  - `h20` 平均绝对差约 `0.272`
  - `h20` 最大绝对差约 `4.490`
- 所以旧 formal `r1 / r2` 里“`v255` 更适合当默认主候选”的结论，已经不再代表当前代码。

### 结果二：在当前代码口径下，静态赢家已经收敛为 `v250`
- 当前静态 `trend_up_low_vol_ml25_none25_v250`：
  - `full_excess_sharpe = 0.759`
  - `weak_window_20250905_20260319_excess_sharpe = 1.073`
  - `trend_up_low_vol_weak_window_20250905_20260319_excess_sharpe = 1.083`
- 当前静态 `trend_up_low_vol_ml25_none20_v255`：
  - `full_excess_sharpe = 0.675`
  - `weak_window_20250905_20260319_excess_sharpe = 0.281`
  - `trend_up_low_vol_weak_window_20250905_20260319_excess_sharpe = 0.094`
- 当前 base-equivalent 默认 artifact：
  - `full_excess_sharpe = 0.294`
  - `weak_window_20250905_20260319_excess_sharpe = 0.151`
  - `trend_up_low_vol_weak_window_20250905_20260319_excess_sharpe = -0.186`
- 这说明：
  - 在当前代码下，旧 live default 已明显落后；
  - `v250` 已经不是“防守候选之一”，而是新的 live default。

### 结果三：动态控制器方向仍成立，但当前还不足以越过新的 live 默认值
- 当前最强 full 端动态候选：
  - `0.791 / 0.992 / 0.980`
- 当前最强 balance 动态候选：
  - `0.775 / 1.013 / 1.008`
- 它们都说明：
  - 动态方案已经开始接近“攻守切换”；
  - 但还没有形成一个足以绕过 live `v250` 默认值、直接升级执行端的单一赢家。

### 本轮结论
1. 旧 formal `r1 / r2` 结论已不再代表当前代码，应整体降级为历史口径。
2. 当前代码下的静态默认值应升级为：
   - `trend_up_low_vol_ml25_none25_v250 @ 504 / 21 / 520`
3. `trend_up_low_vol_ml25_none20_v255` 继续保留为进攻对照，而不是 live default。
4. 动态控制器研究线继续保留，但它的比较基准已经改成 live `v250`，不再是旧 split verdict。

### 当前决策
1. 执行端 wrapper 已切换共享默认参数为：
   - `trend_up_low_vol=ml:0.25,none:0.25,v2:0.50`
2. live artifact 已重训完成：
   - `trained_at = 2026-03-28 15:19:20`
3. 后续所有正式动态比较，默认都要先回答：
   - 能否在同一 formal 口径下跑赢 live `v250`
4. 后续若继续推进动态控制器，不再把“是否沿用旧 `v255` 口径”作为讨论前提。

## 2026-03-28 `advanced_ml (ma50 baseline, lgbm)` `market_features` 正式对照（`legacy_v7` vs `expanded_v24`）
### 本轮目标
- 直接回答：`2026-03-28` 这次底层状态与 `market_features` 扩容后，当前执行端到底是整体更强，还是只是“看起来更现代、但收益反而掉了”。
- 把“怀念旧 `v255` 高收益”和“当前 live `v250` 是否合理”拆成两件事，用同一 execution stack 正式比较。

### 本轮动作
- 新增脚本：
  - `daily_research/baseline/compare_market_feature_profiles.py`
- 正式输出目录：
  - `daily_research/output/market_feature_profile_compare_20260328_formal_r1`
- 固定比较对象：
  - `base_global`
  - `trend_up_low_vol_ml25_none25_v250`
  - `trend_up_low_vol_ml25_none20_v255`
  - `trend_up_low_vol_controller_gap0p024192_vol0p176128_ret100p014717`
- 固定口径继续保持：
  - `liquid500`
  - `next_open`
  - `20190101 -> 20260327`
  - `504 / 21 / 520`

### 结果一：`expanded_v24` 不是整体升级，而是“修 base + 保住 `v250` 防守”
- `base_global`
  - `expanded_v24 - legacy_v7 = full_excess_sharpe +0.278`
  - `weak_window_20250905_20260319_excess_sharpe +0.808`
  - `trend_up_low_vol_weak_window_20250905_20260319_excess_sharpe +0.686`
- `trend_up_low_vol_ml25_none25_v250`
  - `full_excess_annual_return +0.08%`
  - `full_excess_sharpe -0.027`
  - `recent_full_excess_sharpe +0.097`
  - `weak_window_20250905_20260319_excess_sharpe +0.081`
  - `trend_up_low_vol_weak_window_20250905_20260319_excess_sharpe +0.115`
- 这说明：
  - `24-feature` 新系统并不是把所有候选都一起抬高；
  - 它明显修复了旧 `legacy_v7` 下几乎失效的 `base_global`；
  - 对当前 live `v250`，它属于“全样本收益几乎持平、full Sharpe 略低，但最近窗口和弱窗口更强”。

### 结果二：`expanded_v24` 明显吃掉了旧 `v255` 与动态控制器的进攻上沿
- 用同一 execution stack 直接比较时，`expanded_v24 - legacy_v7` 的结果是：
  - `v250`：`full_excess_sharpe = -0.027`，但 `weak_window_20250905_20260319_excess_sharpe = +0.081`
  - `v255`：`full_excess_sharpe = -0.184`，`weak_window_20250905_20260319_excess_sharpe = -0.538`
- 这说明：
  - 新系统不是把所有候选都抬高；
  - 它主要保住并强化了当前 live `v250` 的防守；
  - 但旧 `legacy_v7` 下 `v255` 的进攻上沿，确实比当前 `expanded_v24` 更高。
- `trend_up_low_vol_ml25_none20_v255`
  - `full_excess_sharpe -0.184`
  - `full_excess_annual_return -3.45%`
  - `weak_window_20250905_20260319_excess_sharpe -0.538`
  - `trend_up_low_vol_weak_window_20250905_20260319_excess_sharpe -0.656`
- `trend_up_low_vol_controller_gap0p024192_vol0p176128_ret100p014717`
  - `full_excess_sharpe -0.058`
  - `full_excess_annual_return -0.67%`
  - `weak_window_20250905_20260319_excess_sharpe -0.110`
  - `trend_up_low_vol_weak_window_20250905_20260319_excess_sharpe -0.134`
- 这说明：
  - 你记忆里“以前明显更能打”的感觉，主要不是错觉；
  - 它集中体现在旧 `legacy_v7` 口径下的 `v255` 进攻腿；
  - 当前 `expanded_v24` 不是把整条 frontier 全面右移，而是把收益结构重排了。

### 本轮结论
1. 当前 live `expanded_v24 + v250` 仍然成立，不应该因为怀念旧 `v255` 高收益就直接整体系回滚。
2. 但 `24-feature` 不能再被表述成“整体架构全面增强”；更准确的表述是：
   - `base` 更强；
   - `v250` 防守更稳；
   - `v255` 与当前动态控制器的进攻边更弱。
3. 当前真正的研发目标不再是争论“是否回滚到底层旧系统”，而是：
   - 如何在保住 `v250` 防守改进的前提下，补回 `v255 / dynamic` 的进攻能力。

### 当前决策
1. live 默认继续保持：
   - 当前代码口径
   - `expanded_v24`
   - `trend_up_low_vol_ml25_none25_v250`
2. 下一步优先研究：
   - `attack leg` 的 `market_features` 裁剪或 hybrid profile；
   - 再用同一 formal 口径比较其是否能追回旧 `v255` 的进攻上沿，而不破坏当前 `v250` 的弱窗口防守。
## 2026-03-28 `advanced_ml (ma50 baseline, lgbm)` `market_features` focused pruning 正式扫描（以 `expanded_v24` 为锚）
### 本轮目标
- 在已经确认 `expanded_v24 + v250` 是当前 live 默认之后，不再泛泛争论“要不要整体系回滚”。
- 直接回答更具体的问题：
  - `24-feature` 里有没有一组更小的 profile，能在不破坏当前 live `v250` 防守的前提下，把旧 `v255` 的进攻边拉回来？
- 本轮 focused pruning 只看当前最有信息量的六组 profile：
  - `legacy_v7`
  - `continuous_quadrant_v9`
  - `expanded_state_only_v14`
  - `expanded_no_market_state_v15`
  - `expanded_no_buckets_v18`
  - `expanded_v24`

### 本轮动作
- 新增/扩展：
  - `daily_research/baseline/ml_alpha.py`
    - 把 `market_feature_profile` 从“新旧两档”扩成可复用的 profile registry
  - `daily_research/baseline/compare_market_feature_profiles.py`
    - 新增多 profile anchored comparator 输出
- 正式输出目录：
  - `daily_research/output/market_feature_profile_pruning_20260328_formal_r1`
- 固定口径继续保持：
  - `liquid500`
  - `next_open`
  - `20190101 -> 20260327`
  - `504 / 21 / 520`
  - 本轮先 `--skip-dynamic`，先把静态 offense / live-defense frontier 跑清楚

### 结果一：简单 pruning 没有产生新的单一升级赢家
- 当前锚点仍是：
  - `expanded_v24`
- `trend_up_low_vol_ml25_none20_v255`
  - `legacy_v7`: `full_excess_sharpe = 0.860`, `weak_window_20250905_20260319_excess_sharpe = 0.819`
  - `continuous_quadrant_v9`: `0.594 / 0.765`
  - `expanded_state_only_v14`: `0.544 / 0.389`
  - `expanded_no_market_state_v15`: `0.481 / 0.199`
  - `expanded_no_buckets_v18`: `0.447 / -0.380`
  - `expanded_v24`: `0.675 / 0.281`
- `trend_up_low_vol_ml25_none25_v250`
  - `legacy_v7`: `full_excess_sharpe = 0.786`, `weak_window_20250905_20260319_excess_sharpe = 0.992`
  - `continuous_quadrant_v9`: `0.579 / 0.787`
  - `expanded_state_only_v14`: `0.568 / 0.393`
  - `expanded_no_market_state_v15`: `0.512 / 0.392`
  - `expanded_no_buckets_v18`: `0.505 / -0.068`
  - `expanded_v24`: `0.759 / 1.073`
- 这说明：
  - 没有任何一个中间态 profile 能同时拿到“比 `expanded_v24` 更强的 live `v250` 防守”和“比 `expanded_v24` 更强的 `v255` 进攻”；
  - `expanded_no_buckets_v18` 明显是错误方向；
  - `expanded_state_only_v14` 和 `expanded_no_market_state_v15` 也没有形成有效折中。

### 结果二：`continuous_quadrant_v9` 是最接近可讨论的 pruning，但仍不够
- 相对当前锚点 `expanded_v24`：
  - 对 `v255`：
    - `full_excess_sharpe = -0.082`
    - `weak_window_20250905_20260319_excess_sharpe = +0.484`
    - `trend_up_low_vol_weak_window_20250905_20260319_excess_sharpe = +0.604`
  - 对 live `v250`：
    - `full_excess_sharpe = -0.180`
    - `weak_window_20250905_20260319_excess_sharpe = -0.286`
    - `trend_up_low_vol_weak_window_20250905_20260319_excess_sharpe = -0.371`
- 这说明：
  - 它确实证明“当前 24 维特征里有一部分在压制 `v255` 的弱窗口表现”；
  - 但它并没有把 `v255` 的 full 端追回来，反而还会明显伤到 live `v250`；
  - 所以它只能作为结构线索，不能直接晋级。

### 结果三：真正清晰的 frontier 变成了“双 profile 分工”
- offense 最强 profile 仍是：
  - `legacy_v7`
- live-defense 最强 profile 仍是：
  - `expanded_v24`
- 而且这两边都不是简单 pruning 能统一起来的。
- 这说明：
  - 当前更值得做的，不再是继续扫“单一全局 profile”；
  - 而是把问题改写成：
    - 是否能让 offense leg 用 `legacy_v7`
    - 同时让 defense/live leg 继续用 `expanded_v24`
    - 再放进同一套 formal 攻守控制器里比较

### 本轮结论
1. `expanded_v24 + v250` 继续成立，不回滚。
2. 简单 `market_features` pruning 不是当前最优研发方向。
3. `legacy_v7` 仍然承载最强的 offense edge，但不能直接整体回滚，因为它会削弱当前 live-defense 口径。
4. 下一步正式研发重点应收口为：
   - 双 profile 攻守控制器
   - 而不是继续做单一全局 profile 的裁剪比赛

### 当前决策
1. 执行端默认值不变：
   - `expanded_v24 + trend_up_low_vol_ml25_none25_v250`
2. 后续若继续推进收益上沿，优先做：
   - `legacy_v7` offense leg
   - `expanded_v24` defense/live leg
   - 同口径 formal comparator

## 2026-03-28 `advanced_ml (ma50 baseline, lgbm)` 旧 `legacy_v7` 进攻栈 vs 当前 `expanded_v24` live 栈正式 A/B
### 本轮目标
- 在同一正式长窗口下，把“旧最强 offense 栈”和“当前 live 默认栈”直接做硬 A/B。
- 不再只看“同候选跨 profile delta”，而是直接回答：
  - 当前系统到底是升级了，还是变弱了？

### 本轮产物
- 正式输出目录：
  - `daily_research/output/market_feature_stack_ab_20260328_formal_r1`
- 关键文件：
  - `profile_results.csv`
  - `pairwise_comparison.csv`
  - `summary.md`
  - `stack_ab/summary.md`
  - `stack_ab/verdict.json`
  - `stack_ab/stack_metrics.csv`
- 关键脚本：
  - `daily_research/baseline/compare_market_feature_profiles.py`
  - `daily_research/baseline/render_market_feature_stack_ab.py`

### 固定口径
- `20190101 -> 20260327`
- `liquid500`
- `next_open`
- `lgbm`
- `504 / 21 / 520`
- 本轮只保留静态：
  - `base_global`
  - `trend_up_low_vol_ml25_none20_v255`
  - `trend_up_low_vol_ml25_none25_v250`
- 本轮显式：
  - `--skip-dynamic`

### 结果一：旧 offense 栈 full 端更强，当前 live 栈 recent / weak 更强
- 旧 offense 栈：
  - `legacy_v7 | trend_up_low_vol_ml25_none20_v255`
  - `full_annual_return = 15.54%`
  - `full_excess_annual_return = 18.22%`
  - `full_excess_sharpe = 0.860`
  - `weak_window_20250905_20260319_excess_sharpe = 0.819`
  - `trend_up_low_vol_weak_window_20250905_20260319_excess_sharpe = 0.751`
- 当前 live 栈：
  - `expanded_v24 | trend_up_low_vol_ml25_none25_v250`
  - `full_annual_return = 13.92%`
  - `full_excess_annual_return = 16.57%`
  - `full_excess_sharpe = 0.759`
  - `weak_window_20250905_20260319_excess_sharpe = 1.073`
  - `trend_up_low_vol_weak_window_20250905_20260319_excess_sharpe = 1.083`
- 当前 live 相对旧 offense 的直接 delta：
  - `full_annual_return -1.62%`
  - `full_excess_annual_return -1.65%`
  - `full_excess_sharpe -0.100`
  - `recent_full_excess_sharpe +0.030`
  - `weak_window_20250905_20260319_excess_annual_return +7.28%`
  - `weak_window_20250905_20260319_excess_sharpe +0.254`
  - `trend_up_low_vol_weak_window_20250905_20260319_excess_sharpe +0.332`
- 这说明：
  - 如果标尺是旧 `legacy_v7 + v255` 的 full 端进攻，当前 live 的确更弱；
  - 但如果标尺是当前执行端最需要的 recent / weak / focus-weak 稳健性，当前 live 又是明显更强。

### 结果二：差异主要来自 `expanded_v24` 对 `v255` 的伤害更大，而对 `v250` 基本是“保住 + 加固”
- profile-only change：
  - `v255` 上：`expanded_v24 - legacy_v7 = full_excess_sharpe -0.184`，`weak_window_20250905_20260319_excess_sharpe -0.538`
  - `v250` 上：`expanded_v24 - legacy_v7 = full_excess_sharpe -0.027`，`weak_window_20250905_20260319_excess_sharpe +0.081`
- 同 profile 内部切换：
  - `legacy_v7` 下 `v250 - v255 = full_excess_sharpe -0.073`，`weak_window_20250905_20260319_excess_sharpe +0.173`
  - `expanded_v24` 下 `v250 - v255 = full_excess_sharpe +0.084`，`weak_window_20250905_20260319_excess_sharpe +0.792`
- 这说明：
  - 新 profile 对 `v255` 的 full 和 weak 都伤得很重；
  - 但对 `v250` 则接近“full 基本持平、weak 明显增强”；
  - 当前系统内部的静态偏好，也已经从旧 `legacy_v7` 的 offense 倾向，切到了 `expanded_v24` 的 live-defense 倾向。

### 本轮结论
1. 这次升级不能再被表述成“整体都更强”。
2. 但它也不能被表述成“白改了、全面变弱”。
3. 更准确的结论是：
   - 当前系统牺牲了旧 `legacy_v7 + v255` 的 full 端进攻上沿；
   - 换来了 current live `expanded_v24 + v250` 在 `recent / weak / focus-weak` 三层更强的稳健性。
4. 所以后续研发主线继续收口为：
   - `legacy_v7` offense leg
   - `expanded_v24` defense/live leg
   - 同一 formal 口径下的双 profile 攻守控制器

### 当前决策
1. 执行端默认值继续保持：
   - `expanded_v24 + trend_up_low_vol_ml25_none25_v250`
2. 不整体系回滚到旧 `legacy_v7`。
3. 后续若要追回收益上沿，不再继续泛泛争论“新系统是不是不如旧系统”，而是直接围绕：
   - `legacy_v7` offense leg
   - `expanded_v24` defense/live leg
   - 双 profile formal comparator

## 2026-03-28 历史快照复刻：`2026-03-24` 旧 `lgbm / histgb / etr` 模型族高收益审计
### 本轮目标
- 不再只靠研究日志判断 `887.75%` 是否可信。
- 直接把 `2026-03-24` 的旧模型族对照，在对应历史代码快照上重跑成可审计证据。

### 本轮动作
- 先用当前代码做了两轮探针：
  - `legacy_v7 + current compare_ml_model_families + auto-trim`
  - `legacy_v7 + current compare_ml_model_families + no-auto-trim-history`
- 结果都无法复刻旧日志量级，说明问题不只在 `market_features`，还有脚本实现与状态层的历史漂移。
- 随后改用 `git worktree` 拉起历史快照：
  - commit: `e7d0f8d151c6667220f8ca5d0a6f98ab3b4b075d`
  - commit time: `2026-03-24 18:58:23 +0800`
  - subject: `执行端12`
- 在隔离快照里直接运行当时的原脚本：
  - `daily_research/baseline/compare_ml_model_families.py`
- 为了尽量贴近旧日志，再追加一轮：
  - `--end-date 20260319`
- 审计产物已拷回当前工作区：
  - `daily_research/output/advanced_ml_model_family_compare_20260319_legacy_snapshot_reaudit_r2`
  - `audit_metadata.json`

### 结果一：旧高收益量级被历史快照成功复刻
- `lgbm`
  - `full_excess_total_return = 910.30%`
  - `full_excess_annual_return = 77.57%`
  - `full_excess_sharpe = 2.398`
  - `recent_full_excess_total_return = 118.01%`
  - `recent_full_excess_sharpe = 2.789`
  - `latest_weak_excess_total_return = 4.96%`
  - `latest_weak_excess_sharpe = 0.230`
- `histgb`
  - `full_excess_total_return = 337.24%`
  - `full_excess_sharpe = 1.395`
- `etr`
  - `full_excess_total_return = 322.53%`
  - `full_excess_sharpe = 1.691`

### 结果二：它与旧日志已经足够接近，可以确认旧记录不是伪高收益
- 旧日志记录：
  - `lgbm full_excess_total_return ≈ 887.75%`
  - `lgbm full_excess_sharpe ≈ 2.300`
  - `recent_full ≈ 127.99% / 2.816`
  - `latest_weak ≈ 9.79% / 0.475`
- 当前历史快照复刻：
  - `910.30% / 2.398`
  - `118.01% / 2.789`
  - `4.96% / 0.230`
- 这说明：
  - 旧日志里的高收益不是凭空写出来的假数字；
  - 在旧系统快照里，`lgbm` 确实能跑出“全样本超额总收益接近 9x、Sharpe > 2”的量级；
  - 当前和旧日志之间的细小差异，更像是数据更新时间、TQ 数据回补或环境细节漂移，而不是结论层面的翻案。

### 本轮结论
1. `2026-03-24` 那段旧高收益，在旧系统里是真实结果。
2. 但它属于历史快照系统，不属于今天的 current live 系统。
3. 所以后续表述应固定为：
   - 旧 `lgbm` 高收益是真实历史结果；
   - 但不能直接拿它替代今天的 current live benchmark。

### 当前决策
1. 旧 `887.75%` 不再按“存疑旧日志”处理。
2. 未来若再遇到类似“历史高收益到底真不真”的争议，优先走：
   - 当前代码探针
   - 历史快照复刻
   - 再做 current live 同口径 A/B

## 2026-03-28 用户显式要求执行端切到最高收益后端
### 本轮目标
- 用户已明确要求“我要的就是最高收益”。
- 不再继续把当前代码口径下的 `expanded_v24 + v250` 当作执行默认值。
- 直接把执行端切到已经审计复刻过的旧高收益快照后端。

### 本轮动作
- 新增当前执行 wrapper 的快照后端转发器：
  - `daily_research/execution/high_profit_backend.py`
- 修改当前执行入口：
  - `daily_research/execution/update_model.py`
  - `daily_research/execution/run_trade_plan.py`
- 当前 wrapper 不再直接调用当前工作区 `baseline/train_trade_model.py` 与 `baseline/generate_daily_trade_plan.py`；
  而是转发到历史快照：
  - commit: `e7d0f8d151c6667220f8ca5d0a6f98ab3b4b075d`
  - commit time: `2026-03-24 18:58:23 +0800`
  - local worktree: `H:/new_tdx64/PYPlugins/user_snapshot_codex_e7d0f8d`
- 同时保持落盘位置不变：
  - `daily_research/execution/models/latest_ml_model.joblib`
  - `daily_research/execution/models/latest_ml_model.json`
  - `daily_research/execution/output/latest_trade_plan.txt`

### 结果
- `update_model.py` 已在旧高收益快照后端重训成功：
  - `trained_at = 2026-03-28 22:59:12`
  - `latest_data_date = 2026-03-27`
  - `execution_date = 2026-03-30`
  - `model_family = lgbm`
  - `regime_ma_window = 50`
  - `feature_count = 34 / 34 / 34`（`h5 / h10 / h20`）
  - `state_ensemble_weights = {}`
- `run_trade_plan.py` 已在同一旧快照后端生成完成：
  - 输出目录：`daily_research/execution/output/20260327`
  - 最新建议文件：`daily_research/execution/output/latest_trade_plan.txt`
  - 本次结果：`今日无明确调仓动作`
- 运行过程中虽然仍打印了 TQ `Load DLL ERROR Version:309 / Get PyGILState_* Error` 提示，但旧快照脚本最终完成了模型产物写出与计划生成。

### 本轮结论
1. 当前执行端已经不再是“当前代码口径下的 live-defense 默认值”。
2. 当前执行端已切到“已审计复刻的旧高收益快照后端”。
3. `expanded_v24 + v250`、`legacy_v7 + v255` 与双 profile 控制器，继续保留为当前代码研究侧的比较锚点。

### 当前决策
1. 执行端默认后端切换为：
   - `historical_snapshot_e7d0f8d (ma50 baseline, lgbm) + liquid500 + next_open`
2. 当前 wrapper 继续保留“写回当前 execution 目录”的方式，不直接把整个工作区代码回滚到旧提交。

## 2026-03-28 执行端最高收益回测复核
### 本轮目标
- 直接回测当前执行端实际在跑的旧快照后端，确认它在最新数据 `2026-03-27` 下，是否仍然是当前可确认的最高收益方案。

### 本轮动作
- 运行旧快照 worktree：
  - `H:/new_tdx64/PYPlugins/user_snapshot_codex_e7d0f8d/daily_research/baseline/compare_ml_model_families.py`
- 使用命令：
  - `--model-families lgbm`
  - `--end-date 20260327`
  - `--windows recent_full:20250307:20260327,latest_weak:20250905:20260327`
  - `--experiment-tag advanced_ml_model_family_compare_20260328_execution_backend_livecheck`
- 对照读取当前代码侧今天已经形成的正式结果：
  - `daily_research/output/market_feature_stack_ab_20260328_formal_r1`
  - `daily_research/output/advanced_ml_attack_defense_controller_20260328_formal_r1`
  - `daily_research/output/advanced_ml_model_family_compare_20260328_legacy_v7_lgbm_noautotrim_probe`

### 结果
- 当前执行端实际后端：
  - `execution_backend_snapshot_lgbm`
  - `full_excess_total_return = 958.89%`
  - `full_excess_sharpe = 2.447`
  - `recent_full_excess_total_return = 128.49%`
  - `latest_weak_excess_total_return = 10.01%`
- 当前代码里今天能确认到的几条高收益对照：
  - `legacy_v7_lgbm_noautotrim_probe`
    - `full_excess_total_return = 151.70%`
    - `full_excess_sharpe = 0.882`
  - `legacy_v7 | trend_up_low_vol_ml25_none20_v255`
    - `full_excess_total_return = 110.99%`
    - `full_excess_sharpe = 0.860`
  - `best_dynamic_r1`
    - `full_excess_total_return = 103.60%`
    - `full_excess_sharpe = 0.817`
  - `expanded_v24 | trend_up_low_vol_ml25_none25_v250`
    - `full_excess_total_return = 98.14%`
    - `full_excess_sharpe = 0.759`

### 本轮结论
1. 以 `2026-03-27` 为最新数据日重新回测后，当前执行端实际运行的旧快照后端，仍然是当前可确认的最高收益方案。
2. 它不只是高于当前 live 默认 `expanded_v24 + v250`，也明显高于当前代码侧最强静态进攻腿、动态控制器，以及 `legacy_v7` 的 no-auto-trim probe。
3. 因此截至 `2026-03-28`，把执行端保持在 `historical_snapshot_e7d0f8d (ma50 baseline, lgbm)`，与用户“我要的就是最高收益”的目标一致。

## 2026-03-29 旧快照高收益因果链钉死：`label_gap_off` 受控 ablation
### 本轮目标
- 不再停留在“旧快照收益很高”或“当前代码收益明显更低”的表面现象。
- 用单开关 ablation 直接验证：
  - 旧快照 `958.89% / 2.447`
  - 到底是来自更强 alpha，还是来自 `next_open` 训练边界上的口径问题。

### 本轮动作
- 先做代码考古，确认：
  - 当前与快照 `features.py` 完全一致；
  - 当前与快照 `build_ml_target()` 也一致；
  - 关键差异落在 `daily_research/baseline/ml_alpha.py` 的训练边界：
    - 快照版直接 `train_end_idx = as_of_idx - 1` / `block_start - 1`
    - 当前版新增 `_label_lookahead_bars()`，对 `next_open` 强制回退 `horizon + 1`
- 新建隔离 worktree：
  - `H:/new_tdx64/PYPlugins/user_ablation_labelgap_off`
  - branch: `ablation_labelgap_off_20260328`
- 只做一个改动：
  - 在隔离 worktree 里把 `ml_alpha.py::_label_lookahead_bars()` 临时改成 `return 0`
- 然后重跑与当前 probe 同口径的命令：
  - `compare_ml_model_families.py`
  - `--model-families lgbm`
  - `--market-feature-profile legacy_v7`
  - `--no-auto-trim-history`
  - `--experiment-tag advanced_ml_model_family_compare_20260328_legacy_v7_labelgap_off_ablation`

### 结果
- 当前代码正常 probe：
  - `advanced_ml_model_family_compare_20260328_legacy_v7_lgbm_noautotrim_probe`
  - `full_excess_total_return = 151.70%`
  - `full_excess_sharpe = 0.882`
- 隔离 worktree `label_gap_off` ablation：
  - `advanced_ml_model_family_compare_20260328_legacy_v7_labelgap_off_ablation`
  - `full_excess_total_return = 958.89%`
  - `full_excess_sharpe = 2.447`
- 该 ablation 与旧快照 livecheck 的 full 指标完全对齐：
  - `958.89% / 2.447`
- 训练日志也同步回到旧快照边界：
  - 第一段由当前代码的 `predict_start = 2024-02-06`
  - 回跳为旧快照式的 `predict_start = 2024-01-30`

### 本轮结论
1. 旧快照 `958.89%` 的主因已经被单开关实验证实：
   - 不是 `features.py` 更强；
   - 不是 `build_ml_target()` 公式不同；
   - 而是 `next_open` 训练边界缺少 label-safe gap，存在严重 `label leakage / look-ahead bias`。
2. 这意味着“旧快照高收益”虽然可复刻，但它不是当前执行端可以直接继承的可信 alpha。
3. 当前 wrapper 运行时虽仍指向旧快照后端，但研究判断必须从“最高收益主线”切换为“已识别出真实性问题的历史 artifact”。
4. 这轮也把一个可复用方法沉淀出来：
   - 先做 apples-to-apples 口径对齐
   - 再用隔离 worktree 做 single-switch ablation
- 如果单开关能精确复现旧收益，就先把旧收益视为 artifact 候选，再决定后续执行切回或桥接验证

## 2026-03-29 执行端从旧快照高收益后端切回当前仓安全桥接版

### 背景
- `2026-03-29` 的代码考古与受控 ablation 已钉死：
  - 旧快照 `958.89% / 2.447` 的主因是 `next_open` 训练边界缺少 `label-safe gap`
  - 它属于 `label leakage / look-ahead bias` 产物，不再允许继续作为执行默认后端
- 用户随后明确要求直接处理最现实的问题：
  - 把当前执行端从已证伪的快照高收益后端切回来，或者至少先做一个安全桥接版本

### 实际改动
- 修改执行 wrapper：
  - `daily_research/execution/update_model.py`
  - `daily_research/execution/run_trade_plan.py`
- 不再转发到历史快照 worktree，而是切回当前仓：
  - `daily_research/baseline/train_trade_model.py`
  - `daily_research/baseline/generate_daily_trade_plan.py`
- 恢复执行默认注入：
  - `regime_ma_window=50`
  - `enhanced_profile=up_low_breakout_v2`
  - `trend_up_low_vol=ml:0.25,none:0.25,v2:0.50`
  - `stocks-file=universe/liquid500_latest.txt`

### 验证
- 先实际运行：
  - `python daily_research/execution/update_model.py ...`
- 新模型产物已写出：
  - `daily_research/execution/models/latest_ml_model.json`
  - `trained_at = 2026-03-29 00:36:54`
  - `latest_data_date = 2026-03-27`
  - `model_family = lgbm`
  - `enhanced_profile = up_low_breakout_v2`
  - `state_ensemble_weights.trend_up_low_vol = ml:0.25 / none:0.25 / v2:0.50`
- 随后运行：
  - `python daily_research/execution/run_trade_plan.py ...`
- 新计划已写出：
  - `daily_research/execution/output/latest_trade_plan.txt`
  - 生成时间 `2026-03-29 00:37:40`
  - 信号日 `2026-03-27`
  - 执行日 `2026-03-30`
  - 本次结果：`今日无明确调仓动作`

### 本轮遇到的困难与修正
- 一开始为了省时间，把 `update_model.py` 和 `run_trade_plan.py` 并行跑了。
- 结果计划先消费了旧 artifact，出现了“计划生成成功，但绑定的仍是旧模型”的依赖错位。
- 修正方式：
  - 立刻停止把这类 producer-consumer 步骤当成可并行任务；
  - 在训练完成后顺序重跑一次 `run_trade_plan.py`；
  - 重新确认 `latest_trade_plan.txt` 里的模型训练时间已经更新到 `2026-03-29 00:36:53`

### 结论
1. 当前执行端默认后端已不再指向旧快照高收益 artifact，而是切回当前仓无泄漏的安全桥接口径。
2. 研究侧仍保留：
   - `expanded_v24 + v250` 作为当前代码 live-defense 锚点
   - `legacy_v7 + v255` 作为 offense 锚点
   但它们不再与执行 wrapper 的默认后端混写。
3. 后续若要把执行端从“安全桥接版”继续升级到更贴近当前研究 live 默认值，必须先补做同口径桥接验证，而不是再次直接切到一个高收益表象更强的后端。
## 2026-03-29 Same-Protocol Bridge Validation: execution `v250@260` vs live-anchor `v250@520`
### Objective
- Decide whether the current execution default should stay on the 260-tree safe bridge or be upgraded to the current-code live anchor.

### Bridge evidence
- Reused the formal output directory:
  - `daily_research/output/advanced_ml_retrain_tree_impact_20260327_trainwindow_formal_r2`
- Locked the protocol to the same stack:
  - `expanded_v24 + trend_up_low_vol_ml25_none25_v250`
  - `liquid500`
  - `next_open`
  - `504 / 21`
  - weak window `20250905 -> 20260319`
- Only changed one knob:
  - `lgbm_n_estimators = 260`
  - `lgbm_n_estimators = 520`

### Key bridge result
- `v250 @ 504 / 21 / 260`
  - `full_excess_total_return = 61.81%`
  - `full_excess_sharpe = 0.549`
  - `recent_full_excess_total_return = 23.09%`
  - `recent_full_excess_sharpe = 1.039`
  - `weak_window_20250905_20260319_excess_total_return = 6.86%`
  - `weak_window_20250905_20260319_excess_sharpe = 0.692`
  - `trend_up_low_vol_weak_window_20250905_20260319_excess_sharpe = 0.547`
  - `full_avg_turnover = 0.732`
- `v250 @ 504 / 21 / 520`
  - `full_excess_total_return = 65.96%`
  - `full_excess_sharpe = 0.663`
  - `recent_full_excess_total_return = 19.93%`
  - `recent_full_excess_sharpe = 0.885`
  - `weak_window_20250905_20260319_excess_total_return = 11.38%`
  - `weak_window_20250905_20260319_excess_sharpe = 1.154`
  - `trend_up_low_vol_weak_window_20250905_20260319_excess_sharpe = 1.139`
  - `full_avg_turnover = 0.725`

### Engineering obstacle and fix
- During implementation, found that the execution path could not actually express the winning knob:
  - `daily_research/execution/entrypoint_utils.py` had no `--lgbm-n-estimators` injection
  - `daily_research/baseline/train_trade_model.py` and `daily_research/baseline/generate_daily_trade_plan.py` also lacked this CLI argument
- Fixed by:
  - adding `--lgbm-n-estimators`
  - wiring it into `MLAplhaConfig`
  - making the execution wrapper inject `520`
  - exposing `LGBM Trees: 520` in `latest_trade_plan.txt`
- Found one more audit mismatch after the first retrain:
  - artifact internal `ml_config` already showed `520`
  - outer `latest_ml_model.json` initially did not write `lgbm_n_estimators`
- Patched the meta writer and reran `update_model.py` so the external JSON and the internal artifact now agree.

### Live verification
- Retrained with:
  - `python daily_research/execution/update_model.py --data-source tq --start-date 20210101 --benchmark 000300.SH --regime-max-annual-vol 0.32 --regime-quadrants trend_up_low_vol,trend_up_high_vol --ml-target-horizons 5,10,20 --ml-horizon-weights 5:0.2,10:0.3,20:0.5 --ml-train-window-days 504`
- Final live model artifact:
  - `daily_research/execution/models/latest_ml_model.json`
  - `trained_at = 2026-03-29 09:59:12`
  - `model_family = lgbm`
  - `lgbm_n_estimators = 520`
  - `enhanced_profile = up_low_breakout_v2`
  - `state_ensemble_weights.trend_up_low_vol = ml:0.25 / none:0.25 / v2:0.50`
- Rebuilt the plan after training finished:
  - `python daily_research/execution/run_trade_plan.py --data-source tq --start-date 20210101 --benchmark 000300.SH --holding-count 5 --rebalance-freq 1d --regime-ma-window 50 --regime-max-annual-vol 0.32 --regime-quadrants trend_up_low_vol,trend_up_high_vol --max-style-weight 0.50`
- Final plan artifact:
  - `daily_research/execution/output/latest_trade_plan.txt`
  - `generated_at = 2026-03-29 09:59:28`
  - text now includes `LGBM Trees: 520`
  - result remains `今日无明确调仓动作`

### Conclusion
1. The bridge validation is strong enough to support upgrading the execution default from `v250 @ 504 / 21 / 260` to `v250 @ 504 / 21 / 520`.
2. This is a current-code upgrade, not a return to the invalid old snapshot backend.
3. Future execution-path upgrades must keep the same validation discipline:
   - same protocol first
   - then CLI / wrapper expressibility
   - then artifact + meta + downstream-plan triple verification
## 2026-03-29 Gemini standard mode rollback to background resume, plus brain repair
### Objective
- User judged that the previous "frontend persistent collaboration mode" did not actually save time, because Codex later still called Gemini through a separate background CLI process.
- User asked for two things:
  - switch Gemini back to background mode as the standard way
  - repair and improve the brain so it no longer insists on the outdated frontend-first workflow

### What changed
- Patched `daily_research/tools/gemini_frontend.ps1`:
  - added `default_mode = background_resume`
  - `ask` and `closeout` now work without requiring a running frontend window
  - `status` now reports the default mode explicitly
  - frontend `open / close / status` remains available, but only as optional human interactive mode
- Repaired current-state brain docs so they match the real execution chain and Gemini workflow:
  - `brain/environment_model.md`
  - `brain/procedural_memory.md`
  - `daily_research/brain/environment_model.md`
  - `daily_research/brain/procedural_memory.md`
  - `daily_research/brain/semantic_memory.md`
  - `daily_research/brain/working_memory.md`
  - `daily_research/brain/action_system.md`
  - `brain/brain_manifest.json`
  - `daily_research/brain/brain_manifest.json`

### Verification
- `daily_research/tools/gemini_frontend.cmd status` returned:
  - `default_mode = background_resume`
  - `running = false`
- `daily_research/tools/gemini_frontend.cmd ask -Prompt "Reply with exactly: GEMINI_BACKEND_OK"` returned:
  - `GEMINI_BACKEND_OK`
- `python daily_research/tools/doc_guard.py check` passed after all brain edits

### New pitfall discovered
- Running Gemini closeout through background `--resume latest` can still drag stale historical context into the reply.
- During this turn, Gemini closeout incorrectly mentioned:
  - `v24 feature degradation`
  - `V24-Slim`
  - `execution via snapshot bridge`
  even though the current workspace had already moved to the current-code `lgbm520` live anchor.

### Conclusion
1. The standard Codex-to-Gemini path is now background `ask / closeout / sessions`; frontend is optional only.
2. Closing a visible frontend window is no longer treated as "ending Gemini collaboration", because background `--resume` remains usable by design.
3. If background `latest` brings back stale context, Gemini output must be treated as a second opinion rather than a source of truth; workspace files, artifacts, and backtest outputs stay authoritative.
## 2026-03-29 Cross-profile 攻守控制器正式扫描：`legacy_v7` offense vs `expanded_v24` defense
### Objective
- Keep execution default on the current-code `lgbm520` live anchor.
- Formally test whether a cross-profile attack/defense controller can raise annual return above the current clean static offense frontier while retaining the weak-window/focus-weak defense edge.
- If not, stop lingering in the current `v250 / v255 / controller` parameter space and pivot R&D to a new opportunity set.

### New tool
- Added:
  - `daily_research/baseline/scan_cross_profile_attack_defense_controller.py`
- Purpose:
  - use one market-feature profile for offense scoring
  - use another market-feature profile for defense/live scoring
  - keep the same current-code protocol:
    - `liquid500`
    - `next_open`
    - `504 / 21 / 520`
    - `holding_count = 5`
    - `rebalance_freq = 1d`

### Formal run
- Experiment tag:
  - `advanced_ml_cross_profile_attack_defense_20260329_formal_r1`
- Output directory:
  - `daily_research/output/advanced_ml_cross_profile_attack_defense_20260329_formal_r1`
- Focus state:
  - `trend_up_low_vol`
- Static controls:
  - defense: `expanded_v24 | trend_up_low_vol_ml25_none25_v250`
  - offense: `legacy_v7 | trend_up_low_vol_ml25_none20_v255`

### Key result
- Static defense:
  - `full_annual_return = 13.92%`
  - `full_excess_sharpe = 0.759`
  - `weak_window_20250905_20260319_excess_sharpe = 1.073`
  - `trend_up_low_vol_weak_window_20250905_20260319_excess_sharpe = 1.083`
- Static offense:
  - `full_annual_return = 15.54%`
  - `full_excess_sharpe = 0.860`
  - `weak_window_20250905_20260319_excess_sharpe = 0.819`
  - `trend_up_low_vol_weak_window_20250905_20260319_excess_sharpe = 0.751`
- Best cross-profile dynamic by `full_excess_sharpe` and balance:
  - `trend_up_low_vol_cross_legacy_v7_off_expanded_v24_def_controller_gap0p024192_vol0p176128_ret100p014717_offml25_none22_v253_defml25_none23p5_v251p5`
  - `full_annual_return = 15.49%`
  - `full_excess_annual_return = 18.17%`
  - `full_excess_sharpe = 0.836`
  - `weak_window_20250905_20260319_excess_sharpe = 1.359`
  - `trend_up_low_vol_weak_window_20250905_20260319_excess_sharpe = 1.458`
  - `offense_within_focus = 24.90%`

### Direct answer
- The controller is real:
  - it materially strengthens weak-window and focus-weak defense
  - it improves the balance between offense and defense compared with either static leg alone
- But it still fails the user's upgrade gate:
  - `15.49%` annual return is still below the clean static offense frontier `15.54%`
  - no dynamic candidate dominates both static controls on full and weak-window metrics at the same time

### Conclusion
1. Execution default stays unchanged on the current-code `lgbm520 v250` live anchor.
2. Cross-profile controller work is now a finished frontier-mapping step, not the main active frontier.
3. By the user's explicit rule, R&D should now pivot away from the current `v250 / v255 / controller` parameter space and move to a new opportunity set / new alpha family.
## 2026-03-29 `deep_alpha` 新 alpha 家族正式起跑：最小充分矩阵 `backbone` 阶段完成
### Objective
- Stop continuing the `v250 / v255 / controller` frontier.
- Move the new-alpha search onto the already-defined formal entry:
  - `daily_research/deep_alpha/run_minimal_matrix.py`
- Complete the first real formal stage:
  - `backbone`

### Formal run
- Command:
  - `C:\Users\ASUS\miniconda3\envs\quant\python.exe daily_research\deep_alpha\run_minimal_matrix.py --phase backbone --root-tag deep_alpha_minimal_matrix_20260329_backbone_r1`
- Output root:
  - `daily_research/output/deep_alpha_minimal_matrix_20260329_backbone_r1`
- Windows:
  1. `20230214 -> 20240227`
  2. `20240228 -> 20250313`
  3. `20250314 -> 20260327`

### Practical obstacle and fix
- This stage was long enough that the shell wait timed out before the full stage finished.
- The correct recovery method was:
  - rerun the same `root-tag`
  - let the runner `skip-existing`
  - fill the remaining missing pretrain / finetune windows
  - wait for `stage_backbone_selected.json` instead of treating partial window metrics as a finished stage

### Leaderboard
- `enc-patch__pre-nopre__score-manual__rank-plain`
  - `mean_excess_sharpe = 0.744`
  - `min_excess_sharpe = -0.529`
  - `mean_excess_total_return = 60.76%`
  - `finetune_undertrained_count = 0`
  - `pretrain_undertrained_count = 0`
- `enc-gru__pre-nopre__score-manual__rank-plain`
  - `mean_excess_sharpe = 0.626`
  - `min_excess_sharpe = 0.014`
  - `mean_excess_total_return = 45.78%`
  - `finetune_undertrained_count = 0`
  - `pretrain_undertrained_count = 0`
- `enc-patch__pre-maskedpre__score-manual__rank-plain`
  - `mean_excess_sharpe = -0.224`
  - `min_excess_sharpe = -1.173`
  - `mean_excess_total_return = -13.03%`
  - `finetune_undertrained_count = 0`
  - `pretrain_undertrained_count = 1`

### Conclusion
1. The first formal winner in the new-alpha family is:
   - `patch_transformer + no pretrain + manual + plain`
2. `masked pretrain` does not currently deserve to stay on the default backbone route:
   - it is not the winner
   - it carries one undertrained pretrain window
   - its three-window average is materially worse
3. The next sensible step is now narrow and concrete:
   - continue to `score_head`
   - do not open more backbone branches first

## 2026-03-29 Gemini hallucination escalation rule was productized

### Trigger
- The existing Gemini background flow was too sticky to `latest` session memory.
- When stale context leaked back in, Codex needed an explicit, repeatable escalation path instead of ad hoc prompt tightening.

### Tooling changes
- Patched `daily_research/tools/gemini_frontend.ps1` to support:
  - `-FreshSession`
  - `-Model`
  - `-Escalate`
- `-Escalate` was defined as:
  - force fresh session
  - if no explicit model is provided, default to `gemini-3.1-pro-preview`
- Frontend window mode can now be reopened as an isolation path with:
  - `daily_research\tools\gemini_frontend.cmd open -ForceNew -Escalate`

### Validation
- Verified the new status surface:
  - `default_mode = background_resume`
  - `escalation_mode = fresh_session_plus_model`
  - `default_escalation_model = gemini-3.1-pro-preview`
- Verified fresh-session background call:
  - `daily_research\tools\gemini_frontend.cmd ask -Prompt "Reply with exactly: GEMINI_FRESH_OK" -FreshSession`
  - returned `GEMINI_FRESH_OK`
- Verified pro escalation path:
  - `daily_research\tools\gemini_frontend.cmd ask -Prompt "Reply with exactly: GEMINI_ESCALATE_OK" -Escalate`
  - returned `GEMINI_ESCALATE_OK`
- Tried `closeout` twice after the upgrade:
  - first with `-Escalate`
  - then with a shorter summary and `-FreshSession`
  - both timed out without a trustworthy return, so the fallback rule was recorded: retry once with a shorter summary, then report failure honestly

### Operational rule learned
- Default collaborative path remains background `--resume`.
- First obvious stale-memory / hallucination event:
  - upgrade to `-FreshSession`
- Repeated drift, critical review, or high-risk closeout:
  - upgrade to `-Escalate`
- If context must be visually isolated from the old thread:
  - open a new frontend window with `open -ForceNew -Escalate`
- Gemini remains a second-opinion tool; final truth still comes from the current workspace artifacts and commands.

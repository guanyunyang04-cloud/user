# Traditional Quant Research

本项目用于传统方向的量化方法研究，聚焦可解释、可复现、低复杂度的研究路径。

## 研究边界

- 优先研究传统量化方法：横截面因子、趋势/均值回复、统计套利、组合构建、风险控制和交易成本建模。
- 默认先做小样本、可解释、可复验实验，再进入复杂模型或生产接入。
- 不把单次回测、未扣费回测、样本内结果或 loose latest 产物作为正式结论。
- 深度学习、强化学习和执行策略实验不是本项目主线；需要时先写入研究假设和对照基线。

## 项目结构

- `backtest.py`、`factors.py`、`metrics.py`、`portfolio.py`: 可复用研究函数和轻量基线模块。
- `tests/`: 稳定性与基础行为测试。
- `experiments/`: 实验配置、脚本和结果摘要入口。
- `research_log/`: 研究记录和结论备忘。
- `data/`: 本地数据占位目录，默认不提交原始数据。
- `brain/`: 项目脑区，记录身份、状态、规则、操作入口和证据。

## 第一阶段方向

1. 建立传统因子研究基线：收益、动量、均线偏离、波动率和换手 proxy。
2. 建立最小回测协议：样本切分、手续费、滑点、持仓约束和风险指标。
3. 建立组合构建基线：等权、排名多空、波动率缩放和行业/风格约束占位。
4. 建立研究日志制度：每个实验必须记录数据范围、假设、参数、结果和下一步。

## 默认研究框架

本项目采用从研究问题到可复验结论的完整传统量化研究框架：

```text
研究问题
-> PIT 数据集
-> 样本构造
-> 特征/因子
-> 标签/目标
-> 模型或打分方法
-> loss/训练目标
-> 预测输出
-> 组合构建
-> 回测执行
-> 评估诊断
-> 实验记录
-> 结论分级
```

完整说明见 `brain/references/research_framework.md`。当前优先跑通 `v2 PIT 数据集 -> 基础传统因子 -> 未来收益标签 -> IC/分组收益 -> Top-N 回测 -> 研究日志` 的第一条闭环，再引入复杂模型架构。

## 快速验证

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest traditional_quant_research/tests
```

## v2 数据入口

- 默认研究数据源：Baostock PIT 日频快照，读取接口为 `dataset_v2.py`。
- 行业表缓存：`python -m traditional_quant_research.dataset_builder_v2 fetch-industry --year 2026`。
- 日频指标缓存：`python -m traditional_quant_research.dataset_builder_v2 fetch-metrics --year 2026 --include-metrics`。
- 行业缓存可选 `--industry-frequency daily|month-start`；`daily` 是精确日频查询，`month-start` 是月初 PIT 行业快照向后填充，适合先做全量行业暴露审计。
- 含行业表面板：`load_tradeable_panel(..., include_industry=True)`。
- 含日频指标面板：`load_tradeable_panel(..., include_metrics=True)`，指标表字段为 `turn/pctChg/peTTM/pbMRQ/psTTM/pcfNcfTTM`。
- v2.2 size 面板：`load_tradeable_panel(..., include_size=True)`，可选表字段为 `total_market_cap/float_market_cap/total_share/float_share/free_share`；旧 snapshot 或无表 snapshot 保持兼容。
- v2.2 size ingestion 契约：`size_source.py` 可把 Tushare `daily_basic` 的 `total_mv/circ_mv/total_share/float_share/free_share` 标准化为项目 `daily_size` schema，并支持年度缓存 `cache/daily_size/year=YYYY.parquet`。
- v2.2 size 缓存入口：`python -m traditional_quant_research.dataset_builder_v2 fetch-size --year 2026 --trade-dates 20260601 --symbols 600000.SH,000001.SZ --size-token <token>`；未传 `--trade-dates` 时优先读取 v2 交易日/股票池缓存，适合按年补数。带 `--symbols` 的 smoke 写入 `cache/daily_size/samples/sample=<hash>/`，不覆盖正式年度 cache。
- v2.2 含 size 快照装配：`python -m traditional_quant_research.dataset_builder_v2 assemble --start-date 2016-01-01 --end-date 2026-06-01 --include-size` 会读取年度 `daily_size` 缓存，并在非空时写入 snapshot 的 `daily_size.parquet` 和 manifest 行数字段。
- 当前限制：行业表已全量接入但口径为 `month-start` 前向填充；daily metrics 已完成全量构建、missingness 和语义审计，但估值字段 PIT 发布时间/修订行为仍未证明；真实市值、流通市值和股本仍未接入。
- v2.2 size 来源侦察入口：`python -m traditional_quant_research.experiments.v2_external_size_source_scout --write-research-log`。当前结论是 Tushare `daily_basic` 为第一验证对象，JoinQuant/RQData 为订阅备选，AkShare/efinance 只作公开端点交叉检查；未完成 token/auth/live probe 前保持 proxy-only。
- Tushare 双票 probe 入口：`python -m traditional_quant_research.experiments.v2_tushare_size_probe --symbols 600000.SH,000001.SZ --trade-dates 20260525,20260601 --write-research-log`。probe 会同时输出原始 `tushare_daily_basic_probe.csv` 和标准化 `daily_size_probe.csv/parquet`；当前环境已安装 `tushare=1.4.29`，但未配置 `TUSHARE_TOKEN/TS_TOKEN`，真实 probe 状态仍为 `skipped/auth_missing`。
- `daily_size` 覆盖率审计入口：`python -m traditional_quant_research.experiments.v2_daily_size_audit --write-research-log`。当前 latest snapshot 返回 `daily_size_absent`，size 覆盖率为 `0.0`。
- 候选晋级门禁入口：`python -m traditional_quant_research.experiments.frontier_promotion_gate --write-research-log`。当前 frontier 在扩展到 2017-2026 后已不再失败于样本数 gate，但仍失败于 size、收益/年度稳定性和风格暴露 gate，策略候选数量仍为 `0`。
- frontier 失败归因入口：`python -m traditional_quant_research.experiments.frontier_failure_attribution --write-research-log`。当前共同弱年为 `2017/2018/2022/2023`。
- 弱年 regime 归因入口：`python -m traditional_quant_research.experiments.frontier_weak_year_regime_attribution --write-research-log`。当前弱年主要对应更低的 `breadth_20d_positive_rate`、`market_ret_20d_mean` 和 `breadth_5d_positive_rate`；该结论只用于失败解释，新 regime rule 仍需 fit/eval 分离和完整门禁复验。

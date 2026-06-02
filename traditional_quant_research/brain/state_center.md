# Traditional Quant Research 状态中枢

- 当前状态：新建并已挂载到 workspace 主脑。
- 当前阶段：第一阶段研究可信度加固与基线策略评估已完成。研究范围已定为上证主板 A 股与深证主板 A 股，并剔除创业板、科创板、ST、停牌、退市等标的；v2 PIT 数据集、默认研究框架、第一条基础传统因子研究闭环、v2 数据/标签审计、全周期多 horizon 单因子诊断、基础回测协议升级和阶段决策报告已跑通。
- 已有骨架：`traditional_quant_research/` 核心函数、`tests/` 基础测试、`experiments/` 实验入口、`research_log/` 研究日志、`data/` 数据契约。
- 默认优先级：先构造沪深主板普通 A 股日频 OHLCV+复权因子数据集，再建立收益/因子/组合/回测最小基线。
- 默认研究框架入口：`brain/references/research_framework.md`。
- 第一条闭环入口：`experiments/baseline_first_loop.py`；首个诊断日志：`research_log/2026-06-02_first_loop_baseline.md`。
- 数据/标签审计入口：`experiments/audit_v2_data_labels.py`；审计日志：`research_log/2026-06-02_v2_data_label_audit.md`。审计显示全周期标签可用率较高，但存在极端 `5d/20d` 标签样本，需后续复权/除权核查。
- 全周期单因子诊断入口：`experiments/full_cycle_factor_diagnostics.py`；诊断日志：`research_log/2026-06-02_full_cycle_factor_diagnostics.md`。当前结果仍属 `diagnostic`，下一步需做基础回测协议升级。
- 基础回测协议升级入口：`experiments/backtest_protocol_upgrade.py`；协议日志：`research_log/2026-06-02_backtest_protocol_upgrade.md`。当前小窗口结果表明日频 Top-N 在费率上更稳，周/月频需要更谨慎解释；正式结论仍需结合更长区间和多因子对照。
- 第一阶段决策报告：`research_log/2026-06-02_phase1_decision_report.md`。结论是可以进入多因子诊断与多因子打分基线，但不建议直接进入传统 ML 生产式建模。
- 默认分支纪律：repo-tracked mutation 优先在 `main` 分支执行。

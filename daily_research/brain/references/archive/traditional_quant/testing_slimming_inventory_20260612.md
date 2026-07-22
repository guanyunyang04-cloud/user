# Traditional Quant Research 测试瘦身清单

日期：`2026-06-12`

## 目标

让默认测试闭环保留核心合同信心，同时把研究回归、外部数据源、重数据构建和历史机制保护放到显式 lane。测试不再承担研究日志和实验记录的全部职责。

## 默认保留

这些测试继续留在普通 lane，保护稳定工程合同：

- `test_traditional_quant_core.py`：基础收益、动量、组合现金和主板股票池合同。
- `test_backtest_protocol.py`：调仓日期、Top-N、换手成本合同。
- `test_horizon_backtest.py`：持有期、buffer、执行约束、group cap、exposure penalty 合同。
- `test_multifactor.py`：rank score、rolling IC、低相关、残差化和 prior-fit pruning 合同。
- `test_dataset.py` / `test_dataset_v2.py`：PIT loader、tradeable panel 和兼容旧 snapshot。
- `test_size_source.py`：size schema、离线标准化、cache 语义和 auth-missing skip。

## 显式研究 lane

这些测试默认不进入普通开发闭环，按阶段或显式 nodeid 运行：

- `frontier_*`
- `low_corr_*`
- `*_audit`
- `*_grid`
- `*_rebuild`
- `*_regime*`
- `*_limitup*`
- `*_kama*`
- `personal_*`
- `generalized_strong_event_pool_research`

它们更多保护研究脚手架、候选门禁和实验复现，不应该阻塞普通小改。

## 外部 / 数据重 lane

这些测试只在数据源、size source、snapshot 构建或 release 维护时运行：

- `test_v2_tushare_size_probe.py`
- `test_v2_external_size_source_scout.py`
- `test_v2_free_size_*`
- `test_v2_cninfo_size_event_audit.py`
- `test_dataset_builder*.py`
- `test_v2_daily_size_audit.py`
- `test_v2_industry_size_source_audit.py`
- `test_v2_daily_metrics_audit.py`

真实 token、live probe、长样本 cache 和年度 snapshot 不进入默认 gate。

## 取消机制保护

以下文件不是旧机制回归，而是保护已取消机制不会复活；保留为 `smoke + guard`：

- `test_frontier_personal_paper_tracking_bootstrap.py`
- `test_frontier_personal_paper_tracking_plan.py`
- `test_frontier_personal_paper_tracking_review.py`
- `test_frontier_personal_candidate_lifecycle_registry.py`

## 后续可归档候选

满足以下条件时，可以从 `tests/` 迁入 `brain/references/` 或删去测试，只保留研究日志：

- 只验证某次历史 run 的 CSV/Markdown 文案，而不保护当前入口、schema、gate 或无未来函数。
- 与当前 North Star 无关，且知识中枢已有稳定结论。
- 同一合同已被更低层、更小 fixture 的测试覆盖。
- 需要真实大 cache、外部 token 或长样本执行才能有意义。

归档前必须保留：研究问题、输入数据、关键参数、结论分级、替代测试或清理理由。

# Quant Data Platform 状态中枢

## 当前状态
- 分脑已由主脑 runtime 初始化并注册，body_root 为 `quant_data_platform`。
- 当前项目处于项目化起步阶段：目录骨架已存在，正式 CLI、tests、registry 迁移和 sharded memmap 仍在建设中。
- 默认研究窗口沿用工作区 canonical 决策：`2010-01-01` 起。
- canonical 数据域原则：行情、5 分钟日级特征、复权因子、估值、行业、指数成分、交易日历、股票池和证券状态进入数据基底；财务季报、业绩预告/快报等慢披露数据暂不进入 v1 默认基底。

## 当前接管重点
- 将 `canonical_data/registry` 与 capped validation memmap 迁入或映射到本项目 registry。
- 建立 `qdp` CLI：status、audit、build-bundle、build-intraday-features、build-sharded-memmap、validate-memmap、cleanup dry-run。
- 让旧 `daily_research` 入口优先读取本项目 registry，旧 registry 仅兜底。

## 根目录关系
- `canonical_data/`：过渡资产入口，等待本项目稳定后退为兼容指针。
- `daily_research/`：正式研究与执行消费者，不再长期拥有共享数据平台职责。
- `a_stock_daily_selection/`：当前只有 output，未注册分脑，先标记为待整理旧目录。

## 默认纪律
- repo-tracked mutation 优先在 `main` 分支执行。
- 数据清理必须先 dry-run、再替代指针、再验证，最后才删除。

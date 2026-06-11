# Quant Data Platform 状态中枢

## 当前状态
- 分脑已由主脑 runtime 初始化并注册，body_root 为 `quant_data_platform`。
- 当前项目已具备最小可用 CLI、tests、registry 迁移、coverage/audit、bundle 构建和 memmap 验证入口。
- 默认研究窗口沿用工作区 canonical 决策：`2010-01-01` 起。
- active canonical bundle 已更新为包含估值、行业、指数成分 sidecar 的 `canonical_data_v1`。
- canonical 数据域原则：行情、5 分钟日级特征、复权因子、估值、行业、指数成分、交易日历、股票池和证券状态进入数据基底；财务季报、业绩预告/快报等慢披露数据暂不进入 v1 默认基底。
- 旧 capped validation memmap 仍可做烟测，但它绑定旧 bundle；新 canonical bundle 的全量/分片 memmap 仍待构建。

## 当前接管重点
- 继续把 `build-sharded-memmap` 从 plan scaffold 推进为真实分片 feature/label store。
- 构建当前 canonical bundle 对应的新 capped validation memmap 或 sharded memmap，替代旧 bundle memmap。
- 在验证通过后，再执行旧 parquet / 旧 `forecast_*.dat` 的清理 dry-run 审批链。

## 根目录关系
- `canonical_data/`：过渡资产入口，等待本项目稳定后退为兼容指针。
- `daily_research/`：正式研究与执行消费者，不再长期拥有共享数据平台职责。
- `a_stock_daily_selection/`：当前只有 output，未注册分脑，先标记为待整理旧目录。

## 默认纪律
- repo-tracked mutation 优先在 `main` 分支执行。
- 数据清理必须先 dry-run、再替代指针、再验证，最后才删除。

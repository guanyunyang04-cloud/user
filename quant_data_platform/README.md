# Quant Data Platform

Canonical brain source: `quant_data_platform/brain/identity_layer.md`.


主脑管辖下的共享量化数据平台分脑。

## 定位

`quant_data_platform` 负责工作区级 canonical 数据基底、registry、coverage audit、policy bundle 和共享 memmap 治理。

它不是独立 Git 仓；版本控制仍归 `H:/quant_project` 主脑工作区统一管理。

## 目录

- `brain/`：本项目分脑。
- `src/`：平台代码。
- `configs/`：canonical 数据域与 profile 配置。
- `registry/`：可审计 manifest、registry 和指针。
- `data/`：大数据资产、tmp、memmap、agent runs，默认不进 Git。
- `docs/`：数据契约、memmap 设计和清理策略。
- `tests/`：平台测试。

## 当前原则

- canonical 数据集完整承载行情、5 分钟特征、复权因子、估值、行业、指数成分和交易过滤域。
- 财务季报、业绩预告、业绩快报等慢披露域暂不进入 v1 默认基底。
- 训练 profile 决定用哪些字段；短线核心 profile 可以不使用估值、行业和指数。
- 旧 `canonical_data/` 是过渡入口，后续由本项目接管 registry 与 memmap 治理。

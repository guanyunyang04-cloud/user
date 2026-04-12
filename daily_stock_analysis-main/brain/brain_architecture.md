# Daily Stock Analysis Brain Architecture

## 1. 分脑定位
`daily_stock_analysis-main/brain/` 是 `daily_stock_analysis-main` 的产品型分脑。

它在整个脑网络中的角色是：

- 保存多市场 AI 股票分析产品的稳定认知
- 为 `src / api / apps / bot / data_provider / tests` 提供统一接管入口
- 与 `daily_research` 的正式执行主线保持边界清晰

## 2. 继承的标准合同
`daily_stock_analysis-main` 继承主脑定义的标准附着分脑合同，统一结构与读写顺序以 [brain/brain_architecture.md](H:/new_tdx64/PYPlugins/user/brain/brain_architecture.md) 为准。

因此本文件不再重复展开整套通用模块定义，只保留本项目相对标准合同的特有强调。

## 3. 本项目的特有强调
- `semantic_memory.md`
  - 重点锁定产品入口、模块边界和 body_map
- `working_memory.md`
  - 重点保存当前产品治理目标和近期改动边界
- `action_system.md`
  - 重点保留 `src / api / apps / bot / data_provider / tests` 的进入顺序

## 4. body 进入顺序
本项目的 body 主要从以下区域进入：

- `daily_stock_analysis-main/src`
- `daily_stock_analysis-main/api`
- `daily_stock_analysis-main/apps`
- `daily_stock_analysis-main/bot`
- `daily_stock_analysis-main/data_provider`
- `daily_stock_analysis-main/tests`

默认原则：

- 先接 brain
- 再按 body_map 进入产品代码
- 不把 `README / docs / AGENTS.md / CLAUDE.md` 当成主入口

## 5. 去冗余原则
- 通用分脑结构只在主脑 `brain_architecture.md` 定义一次。
- 本文件只保留产品特有入口、边界和 body 差异。

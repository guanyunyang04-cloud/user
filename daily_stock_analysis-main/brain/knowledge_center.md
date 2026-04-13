# Daily Stock Analysis 知识中枢

## 1. 稳定事实
- `daily_stock_analysis-main` 保存多市场 AI 股票分析产品的稳定认知
- 它服务于 `src / api / apps / bot / data_provider / tests`

## 2. 硬规则
- 先接主脑，再接本分脑，再进入具体 body
- brain 是当前 AI 入口；`AGENTS.md / CLAUDE.md / SKILL.md` 属于兼容资产
- 新增配置项时要同步更新 `.env.example` 和相关说明

## 3. 已验证教训
- 如果仓库级 AI 资产不指向 brain，后续 agent 会优先读错入口
- 如果模块地图和真实目录不一致，接管和维护会持续绕路

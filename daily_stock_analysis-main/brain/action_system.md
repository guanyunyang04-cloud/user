# Daily Stock Analysis Action System

## 1. 系统入口
- `main.py`
  - 分析任务主入口与调度器
- `server.py`
  - FastAPI 服务入口
- `webui.py`
  - Web 启动脚本
- `api/app.py`
  - FastAPI 应用工厂

## 2. 核心 body 分层
- `src/core/`
  - 主流程编排与市场策略
- `src/services/`
  - 业务服务层
- `src/repositories/`
  - 数据访问层
- `src/agent/`
  - agent、skills、tools、orchestrator
- `data_provider/`
  - 多数据源适配
- `bot/`
  - 机器人命令与平台接入
- `apps/dsa-web/`
  - Web 前端
- `apps/dsa-desktop/`
  - Electron 桌面端
- `scripts/`
  - 构建、检查、辅助脚本
- `tests/`
  - pytest 测试

## 3. 当前行动边界
- 这是一个多入口、多界面的产品仓库
- agent 在进入源码前应先根据任务确定边界：
  - 后端
  - API
  - Web
  - Desktop
  - Workflow
  - Docs
  - AI 协作资产

## 4. 推荐进入路径
- 任务属于主分析链路：
  - `main.py` -> `src/core/` -> `src/services/` -> `data_provider/`
- 任务属于 API / Web：
  - `server.py` / `api/app.py` -> `api/` -> `apps/dsa-web/`
- 任务属于桌面端：
  - `apps/dsa-desktop/`
- 任务属于 agent / bot：
  - `src/agent/` 或 `bot/`

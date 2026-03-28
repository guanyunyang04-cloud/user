# Daily Stock Analysis Procedural Memory

## 1. 仓库级技能
### 1.1 目录边界优先
- 后端逻辑优先放在 `src/`、`data_provider/`、`api/`、`bot/`
- Web 前端改动在 `apps/dsa-web/`
- 桌面端改动在 `apps/dsa-desktop/`
- 部署与流水线改动在 `scripts/`、`.github/workflows/`、`docker/`

### 1.2 变更纪律
- 默认稳定性优先于顺手重构
- 新增配置项时同步更新 `.env.example` 和相关文档
- 涉及用户可见能力、CLI/API、部署、通知、报告结构变化时，同步更新对应说明

### 1.3 AI 协作资产
- `AGENTS.md` 是仓库内 AI 协作规则真源
- `CLAUDE.md` 是兼容入口
- 若修改 AI 协作治理资产，执行：
  - `python scripts/check_ai_assets.py`

## 2. 验证矩阵
- 后端验证：
  - `./scripts/ci_gate.sh`
  - `python -m pytest -m "not network"`
  - `python -m py_compile <changed_python_files>`
- Web / Desktop：
  - `npm ci`
  - `npm run lint`
  - `npm run build`

## 3. 分脑写入路由
- 稳定认知写 `semantic_memory.md`
- 当前优先级写 `working_memory.md`
- 运行口径写 `environment_model.md`
- 系统入口与模块分层写 `action_system.md`
- 时间顺序改动写 `episodic_memory.md`

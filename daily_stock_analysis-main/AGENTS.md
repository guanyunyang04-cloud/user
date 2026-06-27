# AGENTS.md

Canonical brain source: `daily_stock_analysis-main/brain/operations_center.md`.

本文件是仓库原生 AI 生态的兼容入口。它不保存第二套项目治理规则，只把外部工具带回本产品分脑和当前代码结构。

## Object Interfaces

### object `product_repository`
`definition`: 股票智能分析系统代码仓库，覆盖后端分析流程、Web 前端、Electron 桌面端、CI、发布脚本和公开文档。
`body_paths`: `main.py`, `server.py`, `src/`, `data_provider/`, `api/`, `bot/`, `apps/dsa-web/`, `apps/dsa-desktop/`, `scripts/`, `.github/`, `docker/`, `tests/`, `docs/`.
`brain_source`: `daily_stock_analysis-main/brain/`.
`method`: 按用户目标进入相关 body path；长期状态、结构边界和兼容入口语义回写到分脑。

### object `ai_compat_entries`
`definition`: `AGENTS.md`, `CLAUDE.md`, `.github/copilot-instructions.md`, `.github/instructions/*.instructions.md`, `.claude/skills/`。
`method`: 作为外部工具入口指向分脑；内容变化时运行 `python scripts/check_ai_assets.py`。
`invariant`: 这些入口是镜像或路径级补充，不独立扩展项目事实。

### object `repo_change_surface`
`definition`: 当前任务实际触碰的代码、配置、文档、工作流或发布面。
`method`: 先读现有实现、配置、测试、脚本、工作流和文档，再做最小相关改动。
`invariant`: 密钥、账号、远端发布、tag、push、merge、用户工作树状态属于外部状态对象；需要用户意图清楚后再改变。

### object `public_behavior_surface`
`definition`: 用户可见能力、CLI/API 行为、部署方式、通知方式、报告结构、文档入口和 changelog。
`method`: 行为变化写到最近的产品文档；配置语义变化同步 `.env.example`；需要公开回溯时更新 `docs/CHANGELOG.md`。

## Procedure Entries

### procedure `enter_product_surface`
`steps`: 理解用户目标；选择相关对象和 body path；读取当前实现与分脑状态；实施最小改动；按改动面验证；说明结果、风险和未验证项。

### procedure `backend_change`
`paths`: `main.py`, `server.py`, `src/`, `data_provider/`, `api/`, `bot/`, `tests/`.
`validation`: 优先 `./scripts/ci_gate.sh`；较小改动可用 `python -m py_compile <changed_python_files>` 加最近的确定性测试。
`notes`: 数据源、fallback、timeout、retry、报告生成、通知、认证、调度和 API schema 变化会扩大兼容性检查面。

### procedure `client_change`
`paths`: `apps/dsa-web/`, `apps/dsa-desktop/`, 桌面构建脚本。
`validation`: Web 改动使用 `cd apps/dsa-web && npm ci && npm run lint && npm run build`；桌面端在可行时先构建 Web 再构建 Electron。

### procedure `workflow_or_release_change`
`paths`: `.github/**`, `scripts/**`, `docker/**`.
`validation`: 选择最接近的本地验证，并说明影响的流水线、发布路径、权限边界和回滚方式。
`invariant`: 自动 tag 当前是 opt-in 语义，commit title 含 `#patch`、`#minor`、`#major` 才触发版本更新。

### procedure `ai_asset_change`
`paths`: `AGENTS.md`, `CLAUDE.md`, `.github/copilot-instructions.md`, `.github/instructions/**`, `.claude/skills/**`.
`validation`: `python scripts/check_ai_assets.py`。
`writeback`: 稳定语义变化先进入 `daily_stock_analysis-main/brain/`，兼容入口只保留镜像或指针。

### procedure `issue_or_pr_review`
`method`: 可复用 `.claude/skills/analyze-issue/SKILL.md`, `.claude/skills/analyze-pr/SKILL.md`, `.claude/skills/fix-issue/SKILL.md`。
`artifacts`: 分析产物保存到 `.claude/reviews/`。
`external_state`: 评论、approve、request changes、merge、关闭 issue、创建 PR、push、tag、commit 等动作只在用户目标明确时执行。

## Pure Functions

- `select_change_surface(task, paths)`: 返回 backend / client / workflow / docs / ai_asset / review 中的相关对象。
- `select_validation(surface, risk)`: 返回能支撑交付结论的最小检查集合。
- `resolve_doc_target(change)`: README 承载入门和高层总览；专题行为、页面交互、配置与排障写入对应 `docs/*.md`；公开回溯写入 `docs/CHANGELOG.md`。
- `activate_external_state_boundary(action)`: 当动作会改变远端、凭据、发布、tag、push、merge 或用户工作树状态时激活确认边界。
- `derive_delivery_summary(change, validation)`: 输出改了什么、为什么这么改、验证情况、未验证项、风险点和回滚方式。

## Command Palette

```bash
python main.py
python main.py --debug
python main.py --dry-run
python main.py --stocks 600519,hk00700,AAPL
python main.py --market-review
python main.py --schedule
python main.py --serve
python main.py --serve-only
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

```bash
pip install -r requirements.txt
pip install flake8 pytest
./scripts/ci_gate.sh
python -m pytest -m "not network"
python -m py_compile <changed_python_files>
```

```bash
cd apps/dsa-web
npm ci
npm run lint
npm run build

cd ../dsa-desktop
npm install
npm run build
```

```bash
gh pr view <pr_number>
gh pr checks <pr_number>
gh run view <run_id> --log-failed
python scripts/check_ai_assets.py
```

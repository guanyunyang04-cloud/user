# Output / Cache 误删恢复边界

日期：`2026-05-31`

## 结论
- `daily_research/output/` 与 `daily_research/cache/` 曾被误删；当前仓库只保留空目录骨架以恢复 manifest 与 integrity 入口。
- 用户明确确认近期 brain 结论仍正确。因此本事件不推翻 `state_center.md` 中的 recent / research / live/default 结论。
- 缺失 payload 影响的是文件级验证、replay、active artifact inspection、explicit run evidence lookup 和 `project_consistency_check.py` health，不等价于实验结论失效。

## 证据边界
- 事实来源：用户在 2026-05-31 明确说明 `daily_research/output/` 和 `daily_research/cache/` 被误删，并保证近期结论正确。
- 当前可用真源：`daily_research/brain/state_center.md`、本目录 dated references、`daily_research/brain/references/evidence_registry.json` 中已登记的文本结论。
- 当前不可假装可用：误删前的 study summary、protocol summary、lake payload、active execution artifact、daily verdict 和其他 output/cache 物理 payload，除非从真实备份恢复。

## 操作边界
- 不得凭 brain 文本手工重造 `daily_research/output/active_execution_strategy.json`。
- 不得把缺失 output/cache 文件写成 failed trial、模型回归、promotion 失败或 live/default 失效。
- 若需要文件级复验，先从真实备份恢复 `daily_research/output/` 与 `daily_research/cache/`，再运行 `current-frontier`、`project_consistency_check.py`、`doc_guard` 和 `integrity_check`。
- 若恢复源不可用，后续回答必须显式区分：`brain-confirmed conclusion`、`user-confirmed recent conclusion`、`file-backed evidence unavailable`。

## 后续恢复检查
```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow current-frontier --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/tools/project_consistency_check.py
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.doc_guard check
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.integrity_check --json
```

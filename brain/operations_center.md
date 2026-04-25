# 主脑操作中枢

## 1. 默认接管入口
- 只接主脑：
  - `python daily_research/tools/brain_bootstrap.py`
- 接主脑并进入目标分脑：
  - `python daily_research/tools/brain_bootstrap.py --child <brain_id>`

## 2. 结构变更顺序
- 先改 `brain/brain_architecture.md`
- 再改 `brain/brain_manifest.json`
- 再改目标分脑 manifest 与区域特化
- 最后改具体中枢正文和兼容别名

## 2.1 文档收口顺序
- 先判断文档内容属于哪个脑：
  - 工作区共性规则写入 `brain/`
  - 项目事实、入口、流程写入对应 `<child>/brain/`
  - 过长但仍需保留的参考材料写入 `<child>/brain/references/`
- 再把 body 顶层 README / 兼容文档改成简体中文索引，明确指向 brain 真源
- 最后运行 `brain_integrity_check.py --json` 与 `doc_guard.py check`

## 3. 写回路由
- 工作区级当前状态写回 `brain/state_center.md`
- 工作区级固定规则与教训写回 `brain/knowledge_center.md`
- 工作区级拓扑写回 `brain/master_brain.md`
- 工作区级治理写回 `brain/governance_layer.md`
- 工作区级方法、环境与守卫入口写回 `brain/operations_center.md`

## 4. 守卫入口
- `python daily_research/tools/brain_integrity_check.py --json`
- `python daily_research/tools/doc_guard.py check`
- `python daily_research/tools/project_consistency_check.py`
- `doc_guard.py check` 已包含主分脑完整性检查；结构变更后仍建议单独跑一次 `brain_integrity_check.py --json` 便于快速定位

## 5. 环境基线
- brain 文档统一使用 UTF-8
- 工具调用优先走显式 Python 路径或显式脚本入口
- 不依赖“当前 shell 恰好已经处于正确环境”的隐性状态
## 长时任务运行纪律
- 工作区级规则：长时训练任务默认后台运行，前台只负责持续监控到完成，不把训练主进程放在前台阻塞交互。
- 前台不做无意义轮询；只有异常、用户询问状态、或任务完成后读取产物时，才进入日志/summary/checkpoint 检查。
- 目标分脑可以补充具体命令形态，但不得违背“后台运行、监控到完成、完成后一次性复盘写回”的原则。

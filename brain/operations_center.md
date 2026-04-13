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

## 3. 写回路由
- 工作区级当前状态写回 `brain/state_center.md`
- 工作区级固定规则与教训写回 `brain/knowledge_center.md`
- 工作区级拓扑写回 `brain/master_brain.md`
- 工作区级治理写回 `brain/governance_layer.md`
- 工作区级方法、环境与守卫入口写回 `brain/operations_center.md`

## 4. 守卫入口
- `python daily_research/tools/doc_guard.py check`
- `python daily_research/tools/project_consistency_check.py`
- 必要时再跑目标分脑自己的额外检查器

## 5. 环境基线
- brain 文档统一使用 UTF-8
- 工具调用优先走显式 Python 路径或显式脚本入口
- 不依赖“当前 shell 恰好已经处于正确环境”的隐性状态

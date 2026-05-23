# ADR: 工作区迁移到 H:\quant_project

日期：2026-05-23

## Summary
- 结论：当前工作区根目录固定为 `H:\quant_project`。
- 结论：`H:\new_tdx64\PYPlugins\user` 还原为通达信插件用户目录，只保留通达信原生/TQ 用户脚本，不再作为 agent 接管、研究、训练或数据平台根目录。
- 边界：本次迁移只改变工作区位置和冷数据归档位置，不改变 `daily_research/output/active_execution_strategy.json`，不启动训练、study、replay、live/default 或 promotion。

## Facts
- 旧工作区曾位于 `H:\new_tdx64\PYPlugins\user`，这会让项目环境与通达信插件目录耦合。
- 当前项目已经完成主脑优先 Brain Platform 重构，agent 接管入口是 `tools.brain.workflow`，不需要通达信插件目录承载脑区。
- `daily_research` 已建立 TDX-free data platform；正式研究主链路不得依赖 `tqcenter.py`、`pytdx` 或 `mootdx`。
- 新工作区 `H:\quant_project` 是后续代码、脑区、data platform、research lake 和文档维护的默认根目录。
- 旧插件目录 `H:\new_tdx64\PYPlugins\user` 只保留通达信原生需要的用户插件文件，例如 `t0_project\tqcenter.py` 及同组 TQ 脚本；不再放置主脑、分脑、研究产物或量化项目工作区。

## Inferences
- 如果后续 agent 继续从 `H:\new_tdx64\PYPlugins\user` 接管，会重新制造环境耦合，并可能误判通达信插件目录为项目真源。
- README、skill 或历史 reference 中出现旧路径时，应先判断是否为历史证据；当前入口、运行命令和新产物必须使用 `H:\quant_project`。

## Assumptions
- 用户希望释放旧备份占用空间，因此冷数据迁移校验后允许删除完整旧备份。
- 冷数据迁移以 `daily_research/output`、`daily_research/cache`、`daily_research/archive` 的可恢复性为重点；未跟踪的临时进程或部分复制目录不得作为新真源。

## Operating Rules
- 后续 agent 必须从 `H:\quant_project` 运行主脑 capsule：
  `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow capsule --task "<task>" --json`
- 旧 `H:\new_tdx64\PYPlugins\user` 不是 workspace root；不要在其中创建 git repo、brain、output、cache 或 archive。通达信原生/TQ 用户脚本可保留在该目录。
- 涉及长时复制、训练或 study 时，使用 PID 绑定等待：
  `Wait-Process -Id <pid> -Timeout 7200`
- 每轮轮询记录已用时间和预计剩余时间；timeout 只代表本轮等待窗口结束，不代表任务失败。

## Validation
- 从 `H:\quant_project` 运行 git、brain capsule、doc guard、integrity check 和 active artifact diff guard。
- 冷数据复制后，对比源备份与新根目录的关键目录文件数和字节数。
- 只有在复制校验与新根目录守卫通过后，才允许删除 `H:\new_tdx64\PYPlugins\user_quant_project_backup_20260523_155851`。

## Next Allowed Actions
- 以 `H:\quant_project` 为唯一当前工作区根继续接管。
- 如发现历史命令引用旧路径，只在复现实验需要时转换到新根目录；不得把旧路径重新写成当前入口。

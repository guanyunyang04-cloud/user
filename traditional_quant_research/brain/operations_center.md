# Traditional Quant Research 操作中枢

- 接管入口：先运行 workspace capsule，再确认路由选中 `traditional_quant_research` 后读取本脑区。
- 默认写代码采用主脑 personal researcher direct-change：内部研究脚本、旧 helper、旧测试若无真实调用证据或证据价值，直接改到当前合约或删除。
- 继承主脑项目任务命名空间：普通读写、短脚本、测试、临时产物、提交和轮询 / 异步任务默认限制在 `traditional_quant_research` profile；其它项目 dirty/output/process 只作摘要报告，不下钻、不复用、不写成本任务证据，除非用户扩展范围或声明 lease。
- 默认验证采用 changed-surface：先运行 selective verification，按 `blocking_commands` 执行本次必须验证的最小命令。

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.selective_verification --paths <changed_paths> --json
```

- `traditional_quant_research/tests` 整包只作为 shared core、schema、数据口径、候选 gate 或维护/收尾扩展验证；普通单文件改动不默认整包跑。
- 项目普通车道只跑非研究/外部测试：`C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m pytest traditional_quant_research -m "not research and not slow and not data_heavy and not external and not benchmark" -q`。`frontier`、`low_corr`、`v2_*_audit`、`personal`、`limitup`、`kama`、`probe` 等实验回归进入 `research` 或 `external` 车道，按 explicit nodeid 或阶段收口运行。
- 测试瘦身清单：`traditional_quant_research/brain/references/testing_slimming_inventory_20260612.md`。取消 paper tracking / lifecycle 的小测试保留为 `smoke + guard`，用于防止旧机制复活。
- 新实验流程：先把假设、证据等级和稳定结论写入 `brain/references/research_log/`；`research_log/` 只作为现有实验脚本兼容副本，必须声明 `Canonical brain source:`。
- 数据接入流程：先更新 `brain/references/data_catalog.md`，确认字段、日期范围、复权口径和 survivorship bias 处理，再按需同步 `data/catalog.md` 入口副本。

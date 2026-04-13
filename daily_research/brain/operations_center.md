# Daily Research 操作中枢

## 1. 项目地图
- 研究协议与当前判决问题：
  - 先看 `state_center.md`
  - 再进 `daily_research/baseline`
- 正式训练与模型实现问题：
  - 进入 `daily_research/deep_alpha`
- recent / verdict / 守卫 / pipeline 问题：
  - 进入 `daily_research/tools`
- live 默认执行与交易计划问题：
  - 进入 `daily_research/execution`
- 产物与归档：
  - 查看 `daily_research/output`、`daily_research/archive`

## 2. 默认操作纪律
- 当前接管默认不从 `episodic_memory.md` 起步
- 任何问题先判定属于 `formal`、`recent`、`learned-control` 还是 `live`
- 任何问题再判定属于研究环、执行环，还是 promotion 边界
- 新要求一旦改变默认链路，必须先统一代码、脚本入口、manifest、trade plan 展示和 brain 文档

## 3. 协议方法
- formal 验证采用滚动窗口协议
- 每个 formal 窗口都必须写清 `train_end / valid_start / valid_end`
- strongest-model recent 现在只承认 `independent_recent_model_as_of_recent_start`
- 默认执行物化必须使用当前可标注最新数据做 `production full-fit`
- 当前统一权重语义是 `research_raw_target_weight`
- 当前统一上限语义是 `follow_research_raw_no_global_cap`
- recent/live 监控负责解释兑现情况，不负责改写 formal winner

## 4. 高频命令
- 以下 `python ...` 示例默认都指向 `conda run -n yolos python ...` 或显式 `yolos` python
- 主脑优先接管：
  - `python daily_research/tools/brain_bootstrap.py --child daily_research --json`
- strongest-model verdict 刷新：
  - `python daily_research/tools/refresh_strongest_model_verdict.py --help`
- recent protocol 串行收尾 / 监控：
  - `python daily_research/tools/recent_protocol_completion_monitor.py --help`
- 默认次日交易计划：
  - `python daily_research/execution/run_trade_plan.py --help`
- v5 successor 全流程：
  - `python daily_research/tools/run_policy_v5_successor_pipeline.py --help`
- 一致性检查：
  - `python daily_research/tools/project_consistency_check.py`
  - `python daily_research/tools/doc_guard.py check`
- 依赖环境真源：
  - `daily_research/environment.yml`
- 环境创建 / 同步：
  - `conda env create -f daily_research/environment.yml`
  - `conda env update -f daily_research/environment.yml --prune`

## 5. 环境基线
- 依赖环境真源统一收口到 `daily_research/environment.yml`
- `daily_research` 任何程序都必须在 `yolos` 环境下运行
- 命令里的裸 `python` 只是一种简写；真实执行必须绑定到 `conda run -n yolos python` 或显式 `yolos` python
- 脚本内部转调也必须显式落到 `yolos` python，不得回退到 `quant` 或当前 shell Python
- 训练一律使用 GPU；没有 CUDA 就视为阻塞
- 不依赖“当前 shell 已激活 conda 环境”的隐式状态
- `t0_project/tqcenter.py` 是工作区内本地依赖，不由 conda 安装
- brain 文档统一使用 UTF-8

## 6. 写回路由
- 当前状态、当前优先级、当前时态、当前 handoff：
  - `state_center.md`
- 稳定事实、规则、教训：
  - `knowledge_center.md`
- 新命令口径、入口变化、环境与流程：
  - `operations_center.md`
- 接管纪律与治理闭环：
  - `governance_layer.md`
- 时间顺序过程和原始证据：
  - `episodic_memory.md`

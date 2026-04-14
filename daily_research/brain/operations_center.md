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
- 连续型组合策略代理问题：
  - 进入 `daily_research/continuous_policy`
- 产物与归档：
  - 查看 `daily_research/output`、`daily_research/archive`

## 2. 默认操作纪律
- 当前接管默认不从 `episodic_memory.md` 起步
- 任何问题先判定属于 `formal`、`recent`、`learned-control` 还是 `live`
- 任何问题再判定属于研究环、执行环，还是 promotion 边界
- 新要求一旦改变默认链路，必须先统一代码、脚本入口、manifest、trade plan 展示和 brain 文档
- `agent` 做本地 Web 控制台联调、截图、前端验收或短期临时服务检查时，默认在同一 PowerShell 会话中用 `Start-Job` 拉起服务
- 这条 `Start-Job` 默认只属于 `agent` 运行口径，不改用户侧公开教程默认

## 3. 协议方法
- formal 验证采用滚动窗口协议
- 每个 formal 窗口都必须写清 `train_end / valid_start / valid_end`
- 实验设计默认按“最高效、最合理”执行：如果某设定可能只在更强模型上有效，可以重开验证，但必须先写清要验证的假设、额外成本和停止条件
- `epoch formal candidate` 默认先给足 `32` epoch 起步预算；若证据还不够，优先沿同一 `experiment-tag / run_dir` 做 `strict resume`，不优先 fresh rerun
- `non-epoch shadow prototype` 只允许 teacher / shadow / ablation，不适用 `32 epoch` 判决线，也不得直接 promotion
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
- execution app 统一入口：
  - `python daily_research/execution/run_execution_app.py --help`
  - `python daily_research/execution/run_execution_app.py tasks`
  - `python daily_research/execution/run_execution_app.py status`
  - `python daily_research/execution/run_execution_app.py doctor`
  - `python daily_research/execution/run_execution_app.py run --task trade-plan -- --candidate-profile active_execution_strategy`
  - `python daily_research/execution/run_execution_app.py resume --job-id <job_id>`
  - `python daily_research/execution/run_execution_app.py tail --job-id <job_id>`
  - `python daily_research/execution/run_execution_app.py web --port 8765`
- execution Web 控制台独立入口：
  - `python daily_research/execution/run_execution_web.py --port 8765`
  - `agent` 本地验收默认后台模板：
    - `$job = Start-Job -Name daily_research_execution_web -ScriptBlock { Set-Location 'H:\new_tdx64\PYPlugins\user'; & 'C:\Users\ASUS\miniconda3\envs\yolos\python.exe' 'daily_research/execution/run_execution_web.py' '--host' '127.0.0.1' '--port' '8765' }`
- execution 使用教程文档：
  - `daily_research/execution/使用教程.md`
- execution 模拟账户入口：
  - 页面：`/account`
  - 数据文件：`daily_research/execution/current_positions.csv`
  - API：`GET /api/account`、`POST /api/account`、`POST /api/account/reset-example`
- continuous_policy 高频入口：
  - `python daily_research/continuous_policy/run_continuous_policy_protocol.py --help`
  - `python daily_research/continuous_policy/train_policy.py --help`
  - `python daily_research/continuous_policy/evaluate_policy.py --help`
  - `python daily_research/continuous_policy/export_action_panel.py --help`
  - `python daily_research/execution/run_execution_app.py run --task continuous-policy-protocol -- --trainer-backend formal_torch_v2 --epochs 32 --resume-mode strict --pool-name liquid500 --label-preset swing_v2 --train-start-date 20240102 --train-end-date 20251231 --eval-start-date 20260102 --eval-end-date 20260413 --shadow-start-date 20260401 --shadow-end-date 20260413`
  - `python daily_research/execution/run_execution_app.py run --task continuous-policy-train -- --trainer-backend formal_torch_v2 --epochs 32 --resume-mode strict --pool-name liquid500 --label-preset swing_v2`
  - `python daily_research/execution/run_execution_app.py run --task continuous-policy-train -- --trainer-backend prototype_gbdt_v1 --pool-name liquid500 --label-preset swing_v2`
  - `python daily_research/execution/run_execution_app.py run --task continuous-policy-evaluate -- --label-preset swing_v2`
  - `python daily_research/execution/run_execution_app.py run --task continuous-policy-export -- --signal-date 20260413`
  - 当前正式 protocol 参考：
    - `formal_liquid500_20260413_r2_swing_v2`：当前收益/Sharpe 最强的 shadow 参考
    - `formal_liquid500_20260413_r2_balanced_v2`：当前平衡版参考
    - `formal_liquid500_20260413_r2_defensive_v2`：当前防守版参考
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
- continuous_policy 的这条 GPU 约束只对 `formal_torch_v2` 生效；`prototype_gbdt_v1` 保持 shadow 原型身份，不计入正式训练合规
- 不依赖“当前 shell 已激活 conda 环境”的隐式状态
- `t0_project/tqcenter.py` 是工作区内本地依赖，不由 conda 安装
- brain 文档统一使用 UTF-8

## 6. execution app 运行时
- 统一运行时目录：
  - `daily_research/output/execution_app`
- 核心落盘：
  - `runtime_state.json`
  - `events.jsonl`
  - `jobs/<job_id>/metadata.json`
  - `jobs/<job_id>/stdout.log`
  - `jobs/<job_id>/stderr.log`
- 默认流程：
  - 用 `tasks` 看任务注册
  - 用 `doctor` 做健康检查
  - 用 `run --task ...` 触发任务
  - 用 `status` / `tail` 监控
  - 用 `resume` 恢复失败或阻塞作业
  - 用 `unlock --force` 清理确认已失效的 stale lock

## 7. Web 控制台
- 技术栈：
  - `FastAPI`
  - `Jinja2`
  - 原生 JS 轮询
- 页面：
  - `/`
  - `/tasks`
  - `/jobs`
  - `/jobs/<job_id>`
  - `/doctor`
  - `/artifacts/trade-plan`
  - `/continuous-policy`
  - `/account`
  - `/guide`
  - `/settings/runtime`
- API：
  - `/api/status`
  - `/api/doctor`
  - `/api/tasks`
  - `/api/jobs`
  - `/api/continuous-policy`
  - `/api/account`
  - `/api/run`
  - `/api/resume`
  - `/api/unlock`
- 本地启动：
  - 默认只监听 `127.0.0.1`
  - 启动前先确保 `yolos` 已同步 `daily_research/environment.yml`
  - `agent` 联调默认：同一 PowerShell 会话内使用 `Start-Job`
  - 用户侧公开启动默认不因这条 `agent` 约定而变化
 - 页面职责：
  - `Tasks` 负责后台触发任务
  - `Jobs / Job Detail` 负责日志、状态与 resume
  - `Continuous Policy` 负责展示连续策略最新 train / evaluate / protocol / export / runtime state
  - `Account` 负责维护模拟现金、持仓与示例重置
  - `Guide` 负责首次上手、常用流程、恢复与高风险提示
  - `Runtime` 负责 stale lock `force unlock`

## 8. 写回路由
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

# Daily Research 状态中枢

快照日期：`2026-04-13`

## 1. 当前接管摘要
- `daily_research` 是当前工作区的正式生产研究与执行主线
- 当前接管不再从 `episodic_memory.md` 起步
- 当前最小完备接管入口是：
  - `identity_layer.md`
  - `state_center.md`
  - `knowledge_center.md`
  - `operations_center.md`
  - `governance_layer.md`
- `2026-04-13` 接管复核已完成：
  - `project_consistency_check.py` 与 `doc_guard.py check` 已通过
  - `active_execution_strategy.json` 仍对齐 `short_expert_monthly_v1 + regoff_k2_5d_ensemble_native_anchor`
  - `operations_center.md` 的 recent / verdict 入口已纠偏到真实脚本
- `2026-04-13` 依赖环境真源已补齐：
  - `daily_research/environment.yml` 已创建并接入守卫
  - 当前 shell 若不在该环境内，运行报错应先判定为环境偏差而不是直接判定代码回归
- `2026-04-13` 统一运行口径已升级：
  - `daily_research` 任何程序都必须在 `yolos` 环境下运行
  - 脚本默认解释器与脚本内部转调不得再回退到 `quant` 或当前 shell Python
- `2026-04-14` 训练纪律纠偏已写回：
  - 正式训练至少从 `32` epoch 起步；不够就沿同一 `experiment-tag / run_dir` 做 `strict resume` 续训
  - 不足 `32` epoch 的短预算不构成完整训练判决，也不优先 fresh rerun
- `2026-04-14` continuous_policy 训练合同已拆分：
  - `prototype_gbdt_v1 = non-epoch shadow prototype`
  - `formal_torch_v2 = epoch formal candidate`
  - 从此不再把 continuous_policy `v1` 树模型误记为满足 `32 epoch / strict resume` 的正式训练
- `2026-04-13` 执行侧已升级为统一应用骨架：
  - 统一入口：`daily_research/execution/run_execution_app.py`
  - 已具备任务注册、运行日志、状态面板、锁、tail、resume、unlock
  - 运行时目录统一落到 `daily_research/output/execution_app`
- `2026-04-13` execution Web 控制台已落地：
  - 启动入口：`daily_research/execution/run_execution_web.py`
  - 也可通过 `run_execution_app.py web` 启动
  - 当前已具备 Dashboard / Tasks / Jobs / Job Detail / Doctor / Trade Plan / Runtime 页面
  - 当前界面文案已统一切到简体中文，并新增 `/guide` 使用教程页与 `daily_research/execution/使用教程.md`
  - `yolos` 已完成 `FastAPI / uvicorn / jinja2` 实装同步
  - 已做真实 smoke test：`/`、`/api/status`、`/api/doctor`、`/api/run`、`/api/resume`、`/api/unlock` 均通过
  - execution job_id 已升级为微秒级唯一 ID，避免同秒连续触发覆盖旧作业
- `2026-04-13` continuous_policy 并行研究栈已落地：
  - 新目录：`daily_research/continuous_policy`
  - 已具备 `state_builder / label_builder / portfolio_simulator / model / train_policy / evaluate_policy / export_action_panel / run_continuous_policy_protocol`
  - 已接入 execution app task registry 与 Web 控制台 `/continuous-policy`
  - 当前定位是 shadow 连续策略代理，不静默替换 live 默认执行桥

## 2. 当前状态
- 当前统一权重语义：
  - `research_raw_target_weight`
- strongest-model 当前三层答案已经重新对齐：
  - `formal winner = short_expert_monthly_v1`
  - `recent winner = short_expert_monthly_v1`
  - `promotable winner = short_expert_monthly_v1`
- strongest-model 当前 recent protocol batch tag：
  - `short_alpha_recent_model_protocol_20260412_r1`
- strongest-model 当前 recent winner root：
  - `daily_research/output/short_alpha_recent_model_protocol_20260412_r1__short_expert_monthly_v1`
- 当前 live 默认执行：
  - `short_expert_monthly_v1 + regoff_k2_5d_ensemble_native_anchor`
- 当前 execution app 统一入口：
  - `python daily_research/execution/run_execution_app.py status`
- 当前 execution Web 控制台入口：
  - `agent` 本地验收默认口径：在同一 PowerShell 会话内用 `Start-Job` 后台启动 `run_execution_web.py`
  - 这条默认只作用于 `agent` 联调 / 截图 / 验收，不改用户侧公开教程默认
  - 前台调试入口：`python daily_research/execution/run_execution_web.py --port 8765`
- 当前 execution Web 控制台已验证能力：
  - 页面可打开
  - `trade-plan --help` 可经 Web API 后台触发
  - Job Detail 可跟踪日志并执行 resume
  - Runtime 页面可做 stale lock `force unlock`
  - Dashboard / Guide / Runtime 页面布局已做桌面端错位修复
  - `/account` 页面可直接前端维护模拟现金与持仓，并写回 `daily_research/execution/current_positions.csv`
- 当前 continuous_policy 已验证能力：
  - `train_policy.py` 可在 `yolos` 下基于真实市场数据完成 smoke train
  - `formal_torch_v2` 现已支持：
    - `GPU only`
    - `>=32` epoch 起步
    - 同一 `run_dir` 的 `strict resume`
    - `checkpoint_last.pt / checkpoint_best.pt / training_diagnostics.json`
  - `evaluate_policy.py` 可输出连续策略 / teacher 上限 / active manifest / policy_v5b 参考摘要，并附带连续性指标
  - `export_action_panel.py` 可读取当前模拟账户并导出 `daily_live_action_panel.csv + daily_execution_reasoning.json + portfolio_state.json`
  - `run_continuous_policy_protocol.py` 可串联 `train -> evaluate -> shadow continuity -> export`
  - Web `/continuous-policy` 页面可轮询展示最新训练、评估、协议、导出与组合 runtime state
  - `2026-04-14` smoke `formal_torch_v2` 协议 `smoke_cp_v2_protocol_20260414_r2` 已真实跑通：
    - 训练设备 `cuda`
    - `completed_epochs = 32`
    - `best_epoch = 32`
    - 当前仍是 `shadow_only`
  - `2026-04-13` 首轮正式协议 `formal_liquid500_20260413_r1` 已完成，证明 protocol / continuity / shadow 闭环可跑通
  - `2026-04-13` 第二轮 lifecycle preset formal comparison 已完成：
    - `formal_liquid500_20260413_r2_balanced_v2`：年化 `0.3820` / Sharpe `1.6199` / 最大回撤 `-0.0916`
    - `formal_liquid500_20260413_r2_swing_v2`：年化 `0.4157` / Sharpe `1.8885` / 最大回撤 `-0.0941`
    - `formal_liquid500_20260413_r2_defensive_v2`：年化 `-0.1281` / Sharpe `-0.8824` / 最大回撤 `-0.1063`
    - `swing_v2` 当前是 r2 三预设里收益/Sharpe 最强的 shadow 参考
    - `defensive_v2` 的 `open/reduce/exit` 行为指标更强，但收益转负
    - 三个 r2 preset 全部仍是 `shadow_only`，promotion gate 继续关闭
- 当前 learned-control 主锚点：
  - `deployable = short_expert_policy_v5b__k1_20d = 0.1177`
  - `recent = short_expert_policy_v5b = 0.1200`
  - `active v5 fresh formal best = short_expert_policy_v5b = 0.0839`
  - `historical cross-family fresh formal best = short_expert_policy_v4b = 0.1008`

## 3. 当前主问题
- strongest-model 主线已经收口，不再是当前阻塞点
- 当前真正的主问题是：
  - `policy_v5b` 的快桥脆弱性已确认
  - `policy_v5d / policy_v5e` 首轮 successor 未能同时保住 formal / constrained / recent
  - 固定执行桥仍然不是“逐票连续生命史驱动”的执行代理
  - continuous_policy 已从“缺骨架”进入“行为质量不足”阶段：
    - `swing_v2` 已把 `open/reduce/exit` 提升到 `0.4457 / 0.5315 / 0.5714`
    - 但共同卡点仍然是 `hold_share` 近乎为 `0`、`cash_timing_quality_1d` 只有 `0.0077` 量级、`avg_turnover` 仍高于 active manifest
    - 当前更像“会开/加/减/退”的代理，还不是“会稳定持有与留现金”的高手型连续组合代理
    - 新 `formal_torch_v2` smoke 也已经证明：训练合同问题已基本解除，但行为质量问题仍在
    - `smoke_cp_v2_protocol_20260414_r2` 当前 promotion gate 失败项已收缩到：
      - `reduce_success_rate_5d`
      - `cash_timing_quality_1d`
      - `hold_share`
  - 下一轮需要继续以 continuous_policy 为主线，但目标已从“落地骨架”切到“提升行为质量并缩小与参考链路差距”

## 4. 当前优先级
- 保持 `policy_v5b` 作为 learned-control 主研究锚点
- 冻结 live 默认执行，不做静默切换
- 保持 continuous_policy 为并行 shadow 主线，默认优先走 `run_continuous_policy_protocol.py`
- 实验口径已从“默认窄实验”调整为“默认最高效、最合理实验”：若旧设定可能只在更强模型上成立，允许重开，但必须先给出明确假设与收益/成本判断
- continuous_policy 下一轮先修：
  - 开仓质量
  - 减仓 / 退出及时性
  - 现金时机判断
- 把 `policy_v5d / policy_v5e` 明确记为已证伪首轮 successor
- 下一轮只做 `1-2` 个新 successor，且必须显式区别于本轮平滑思路
- 每轮实验后立即写回中枢与证据库

## 5. 当前边界
- formal、recent、promotion、live 不能混写
- recent 验证是 strongest-model verdict 的必备伴随证据
- `requested_recent_end_date` 与 `effective validation end` 必须分开叙述
- 默认执行写入前仍必须走 latest-data `production full-fit`
- 当前 recent 窗口按最近一年 `12` 个月定义
- 正式训练不足 `32` epoch 不构成完整判决；预算不够时优先 `strict resume`，不优先 fresh rerun
- `continuous_policy prototype_gbdt_v1` 只属于 `non-epoch shadow prototype`，不得直接 promotion
- `continuous_policy` 若要进入 promotion 讨论，必须使用 `formal_torch_v2`
- continuous_policy 当前只允许 shadow 训练、评估、影子导出，不允许静默 promotion 为 live 默认
- continuous_policy 只有在正式协议、参考对照、连续 shadow continuity 三者都稳定后，才允许进入 promotion 讨论
- broad hand-crafted repair、broad backbone、Mamba、TSFM、RL 不进入当前正式主线

## 6. 当前时态
- `Past`
  - strongest-model 当前叙事已从旧 `baseline_current` 口径切回真实 winner 口径
  - `policy_v5b bridge sensitivity audit` 已完成
- `Present`
  - learned-control 当前锚点是 `policy_v5b`
  - successor 首轮 `v5d / v5e` 已证明不足
- `Future`
  - 下一轮优先排查 bridge-speed sensitivity / slow-fast consistency
  - learned-control 新分支必须同时跑 `formal + constrained formal + recent`
  - continuous_policy 下一轮优先提升开仓 / 减仓 / 退出 / 现金时机质量，而不是继续只看能否跑通 protocol

## 7. 当前风险
- `policy_v5b` 容易被误读成已经应当替换 live 默认
- 如果只看 constrained formal，会低估 fresh-formal gap
- 如果只看 fresh formal，又会错过 deployable 优势
- 如果不持续写回中枢，新 agent 仍可能沿用旧叙事
- 如果后续环境升级只改本机、不改 `daily_research/environment.yml`，依赖真源还会再次漂移
- 如果程序入口继续跟随当前 shell Python 而不是 `yolos`，环境漂移会被误判成代码问题

## 8. 推荐下一步
- 继续以 `policy_v5b` 作为 learned-control 主锚点
- 冻结 `policy_v5d / policy_v5e`，不进入 promotion 讨论
- 保持 `formal_liquid500_20260413_r2_swing_v2` 作为 continuous_policy 当前最强 shadow 参考，但不进入 promotion
- 下轮优先围绕 `hold_share / cash_timing_quality_1d / avg_turnover` 修正 teacher 标签、动作头与组合分配逻辑
- 再跑下一轮正式 protocol，与 active manifest 和 `policy_v5b` 做同窗对照
- 下一轮 keep gate 继续使用：
  - `constrained formal >= 0.110`
  - `recent >= 0.110`
  - `fresh formal > 0.0839`

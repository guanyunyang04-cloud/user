# Daily Research 知识中枢

## 1. 稳定事实
- `daily_research` 同时负责研究、formal 验证、recent 验证、production full-fit、live 执行和接管治理
- strongest-model research winner、deployable learned-control、live mainline 必须显式区分
- `daily_research/environment.yml` 现在是依赖环境真源
- `daily_research/execution/run_execution_app.py` 现在是执行侧统一应用入口
- `daily_research/execution/run_execution_web.py` 现在是执行侧本地 Web 控制台入口
- 执行侧运行时状态、事件、锁与作业日志统一落到 `daily_research/output/execution_app`
- `daily_research/continuous_policy/` 现在是并行的连续型组合策略代理研究栈
- continuous_policy 的训练、评估、导出产物统一落到 `daily_research/output/continuous_policy`
- `daily_research/continuous_policy/run_continuous_policy_protocol.py` 现在是 continuous_policy 的正式高层协议入口
- continuous_policy 当前已拆成两类训练合同：
  - `prototype_gbdt_v1 = non-epoch shadow prototype`
  - `formal_torch_v2 = epoch formal candidate`
- continuous_policy 正式 protocol 当前支持 `balanced_v2 / swing_v2 / defensive_v2` 三种 lifecycle label preset
- `formal_liquid500_20260413_r2_swing_v2` 当前是 continuous_policy r2 三预设里收益/Sharpe 最强的 shadow 参考
- execution Web 控制台不是纯脚手架，已经在 `yolos` 下做过真实页面 / API / 后台任务联调
- execution Web 控制台当前用户界面与使用教程统一使用简体中文
- execution Web 控制台当前已具备 `/account` 模拟账户页，可直接维护 `daily_research/execution/current_positions.csv`
- execution Web 控制台当前已具备 `/continuous-policy` 页面，可直接查看连续策略的最新 train / evaluate / protocol / export 摘要
- 当前 strongest-model 稳定结论：
  - `formal = short_expert_monthly_v1`
  - `recent = short_expert_monthly_v1`
  - `promotable = short_expert_monthly_v1`
- 当前 live 默认执行稳定语义：
  - `short_expert_monthly_v1 + regoff_k2_5d_ensemble_native_anchor`

## 2. 硬规则
- 必须 `brain-first`
- 默认接管先读 `identity -> state -> knowledge -> operations`
- formal 验证采用滚动窗口协议
- formal 证据只能来自 formal 协议，不得用 production full-fit 回填
- recent 验证现在是 strongest-model 研究闭环必备伴随证据
- 当前统一权重语义是 `research_raw_target_weight`
- 当前统一上限语义是 `follow_research_raw_no_global_cap`
- 默认做最高效、最合理的实验；允许为了验证“某设定是否只在更强模型上成立”而扩实验，但必须先写清假设、成本边界与停止条件，不做无目的广扫
- `epoch formal candidate` 至少从 `32` epoch 起步；不够就沿同一 `experiment-tag / run_dir` 做 `strict resume`，不优先 fresh rerun
- `non-epoch shadow prototype` 不适用 `32 epoch` 口径，但必须显式标记为 shadow-only，不得冒充 promotable formal 结果
- 依赖环境必须先写入真源文件，再谈“环境基线已满足”
- `daily_research` 任何程序都必须在 `yolos` 环境下运行
- 脚本默认解释器与内部 subprocess 统一收口到 `yolos`，不得回退到 `quant` 或当前 shell Python
- 新执行能力优先注册到 execution app task registry，而不是继续追加孤立脚本入口
- 连续策略研究也必须先接 execution app task registry，再谈 Web、shadow export 或 promotion
- continuous_policy 的正式运行默认优先走 `run_continuous_policy_protocol.py`，而不是人工拼接 `train -> evaluate -> export`
- continuous_policy 若要进入 promotion 讨论，默认必须切到 `formal_torch_v2`，并满足 GPU / >=32 epoch / strict resume 的正式合同
- execution 侧默认通过统一应用入口运行、监控、恢复；直接裸跑底层脚本只应用于调试或局部排障
- execution Web 控制台基于 FastAPI + Jinja2，本地只监听 `127.0.0.1`
- `agent` 在本地联调 Web 控制台或短期临时服务时，默认使用同一 PowerShell 会话内的 `Start-Job` 后台方式，而不是 `Start-Process`
- 正式训练前台窗口限时统一为 `10` 小时

## 3. 已验证教训
- recent 胜利不能直接当 promotion 结论
- family 扩宽后，runner 必须同步扩宽
- detached 长实验会降低接管可靠性
- hand-crafted repair 会出现明显天花板
- selected profile 与 constrained best 分裂必须显式建模
- requested recent cutoff 不等于 effective validation end
- strongest-model recent 口径修正后，主问题会整体重排
- external cap 包装与温和平滑 successor 不足以修复 `policy_v5b` 的快桥脆弱性
- 只在 brain 里写“环境基线”而不把依赖环境物化成真源，最终会退化成隐式环境依赖
- 只把部分流程锁到 `yolos` 而放任其他脚本跟随当前 shell Python，最终仍会退化成环境漂移
- execution job_id 如果只精确到秒，在 Web/CLI 连续触发时会覆盖旧作业；运行时 ID 必须保证真正唯一
- 页面如果直接复用通用 `surface-grid` 而没有给 Guide / Runtime / Account 这类页面补明确列布局，桌面端会退化成窄列错位
- `Start-Job` 绑定当前 PowerShell 会话，所以这条默认只属于 `agent` 联调口径，不应误写成用户侧公开启动默认
- 连续策略如果不把 train / evaluate / export / runtime state 的输出约定统一收口，最后会退化成另一套不可接管的孤立脚本族
- 连续策略即使已有 protocol，如果不把连续性指标和参考对照一起固化，仍然会退化成“只看收益数字”的假进展
- 连续策略 export 若盲信旧 runtime state，会在回放旧 `signal_date` 时被未来 runtime 污染；当 runtime 时态晚于目标信号日时必须回退到账户快照重建
- lifecycle preset 会形成清晰的行为-收益权衡：`swing_v2` 当前更接近收益最优，`defensive_v2` 当前更接近行为保守，但可能把收益打成负值
- 当前 continuous_policy 的共同短板已经收缩到两件事：`hold_share` 近乎为 `0`，以及 `cash_timing_quality_1d` 仍然很弱

## 4. 文档边界
- `identity_layer.md`
  - 只保留项目使命、北极星、硬约束与禁区
- `state_center.md`
  - 只保留当前状态、当前问题、当前优先级与 handoff 摘要
- `knowledge_center.md`
  - 只保留稳定事实、硬规则与教训
- `operations_center.md`
  - 只保留地图、环境、命令、流程和写回入口
- `governance_layer.md`
  - 只保留治理闭环与接管纪律
- `episodic_memory.md`
  - 只保留时间顺序过程与原始证据

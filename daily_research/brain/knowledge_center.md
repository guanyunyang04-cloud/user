# Daily Research 知识中枢

## 1. 稳定事实
- `daily_research` 同时负责研究、formal 验证、recent 验证、production full-fit、live 执行和接管治理
- strongest-model research winner、deployable learned-control、live mainline 必须显式区分
- `daily_research/environment.yml` 现在是依赖环境真源
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
- 默认只做窄实验，不做无边界广扫
- 正式训练默认从 `32` 起步；不够就 `strict resume`
- 依赖环境必须先写入真源文件，再谈“环境基线已满足”
- `daily_research` 任何程序都必须在 `yolos` 环境下运行
- 脚本默认解释器与内部 subprocess 统一收口到 `yolos`，不得回退到 `quant` 或当前 shell Python
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

# Daily Research Procedural Memory

## 1. 作用
- 本文件只记录“怎么做”。
- 单次实验过程统一写入 `episodic_memory.md`。
- 稳定事实写入 `semantic_memory.md`。
- 当前判决写入 `working_memory.md`。

## 2. 文档维护规则
- `semantic_memory.md` 只保留长期稳定边界与当前稳定真相。
- `project_map.md` 只保留项目结构、主线地图与决策闭环。
- `working_memory.md` 只保留当前判决、优先级与下一步。
- `action_system.md` 只保留当前高频操作入口。
- 遇到新证据时：
  - 先更新 `working_memory.md`
  - 证据稳定后再沉淀进 `semantic_memory.md`
  - 过程细节进入 `episodic_memory.md`
- 如果 active strategy、execution policy、股票池主线或当前瓶颈发生变化：
  - 同步更新 `working_memory.md`
  - 同步更新 `project_map.md`
  - 必要时同步更新 `semantic_memory.md`
  - 如影响命令入口，再同步更新 `action_system.md`

## 3. 目标函数规则
- 默认目标函数是“执行后净收益最大”，不是“raw holdout 指标最大”。
- 如果用户没有明确改目标函数，禁止用“更稳”替代“更赚钱”。
- 用户明确要求后续改动“不要追求最小，而要追求最有效”后，默认执行原则改为：
  - 优先选择最可能改变 formal / gate / production 结论的改动
  - 不为了保持 diff 小而拆成过多低收益微调
  - 若存在“局部小修”和“更有效但更大”的两个方案，默认优先后者
- 如果用户给出明显高于当前前沿的收益目标：
  - 先把它记为 north star
  - 再把当前 formal / production gate 与长期北极星拆开
  - 不得直接把 aspirational target 改写成当期默认晋级门槛
- execution policy 本身属于研究对象，而不是固定后置适配器。
- 禁止把新研究候选强行塞回旧默认执行壳，再据此宣称“执行端不支持”。
- 如果目标是“净收益最大”，execution policy 的 formal 复核默认优先按 `excess_annual_return` 排序，而不是先用保守 profile 锁死执行层。
- 任何执行升级讨论必须回到：
  - 当前代码
  - 当前特征空间
  - 当前执行桥接
  - 当前现实成本
- 如果目标是把模型练成“短线高手”，默认优先顺序是：
  - 先改 `prediction_horizons / task_loss_weights / score_horizon_weights`
  - 再加 `event_breakout / event_clean_breakout` 与 ranking/listwise loss
  - 再扩 `short_alpha_features` 与 score head
  - 最后才讨论更深 / 更复杂的 encoder
- 如果当前已有 candidate 已经在 latest window 上显示出明显 uplift：
  - 优先先把它推进到 full-budget multi-window formal
  - 再决定是否继续加新特征 / 新 loss / 新输出头
  - 不默认在 candidate 尚未 formal 化前继续堆更多旁支实验
- 如果要做更大更强的 recipe bundle（例如同时改特征、loss、状态惩罚）：
  - 必须先作为独立 opt-in profile 落地
  - 并在补齐预算后决定它是否真的有效
  - 若 budget-stable 后仍输当前主候选，后续默认拆成可归因 ablation，而不是继续叠更多改动
- 如果 short-line expert 要尝试更复杂的输出头：
  - 先注册成独立 opt-in profile（例如 `*_scorehead_*`）
  - 不得直接覆盖已经验证过的 `ridge` 主候选
- 如果修改了 `short_alpha_features` 的定义，必须同步 bump `run_deep_alpha_research.py` 里的 feature cache version，避免旧 feature cache 混入新实验

## 4. formal 判决规则
- formal holdout 负责研究 winner 判决。
- production full-fit 负责默认执行。
- production 结果不得回填为 formal 证据。
- liquid500 默认执行升级判决与 liquid800 / mainboard 研究判决必须分开写。
- 不同股票池、不同成本口径、不同 gate 协议下的结果，禁止直接混成单一“全项目最高”结论。
- 如果用户要“当前全项目最高净收益”，必须新增一层：
  - global deployable leaderboard
  只有进入同一成本引擎、同一 execution-policy audit 搜索空间后，才允许跨 universe 统一排序。
- 如果模型 winner 已稳定、但 execution policy 仍有疑问，可先在同一 raw panel 上做 execution policy audit，再决定是否需要重训模型。

## 5. 训练预算规则
- rich experiment 之前必须先冻结 family epoch budget manifest。
- family epoch budget 的真源固定为 `daily_research/output/deep_alpha_family_epoch_budget_latest.json`。
- 当前冻结预算：`baseline -> 4`，`structure -> 12`，`short_alpha -> 24`，`dynamic_graph -> 16`。
- formal runner 默认读取 manifest，不手填统一 `--epochs`。
- 除非是在当前回合内为了快速探针验证可行性，否则不要主动把 native family budget 压低到 `e4` 之类的临时预算。
- 用户已经明确要求“不要吝啬训练资源、不要轻易中断实验”后，默认动作是：优先跑 native family budget，非硬错误不轻易中断长实验，`e4` 结果只记为 probe/smoke evidence。
- 如果补预算后某窗从 `undertrained` 变成 `stable`，但核心指标明显退化：
  - 必须把它记为“预算补齐后真实泛化边界暴露”
  - 不得继续把中间阶段更亮眼的结果当作最终可推广证据，也不得再用“可能还没训够”掩盖该窗口已经稳定后的退化

## 6. frontier 校准规则
- frontier 默认先跑 `4 / 8 / 12 / 16`。
- 如果最优点落在最右边界，或右边界仍有 objective-aligned budget pressure，再扩到 `24 / 32`，必要时继续到 `40 / 48 / 64`。
- family-default calibration windows 只用于第一阶段。
- 如果某窗在 `32` 仍存在 `objective_aligned_budget_pressure`，但到 `48` 已稳定且结论回撤：
  - 默认停止继续盲目加预算，将其归类为“该窗口对当前 recipe 不形成 clean promotion evidence”
  - 后续优先转向更有效的 recipe/feature/loss 改动，而不是继续单纯加 epoch
- dynamic_graph 允许使用更晚 calibration windows，因为天然有效样本起点更晚。
- 只有在出现以下任一信号时，才允许把“没训够”当作正式判断：`selected_in_tail = true`、`selected_at_right_boundary = true`、`still_improving = true`、`objective_aligned_budget_pressure = true`。
- 如果某条线在 `e4` 探针和 native family full-budget 下给出相同的 selected checkpoint 与同结论结果，则在该窗口上记为 budget-stable，不再把分支优劣归因于 epoch 不足。

## 7. runner 公平性规则
- specialized formal runner 做家族比较时，必须给每个 profile 解析自己的 native family budget。
- 禁止以下错误：
  - `baseline_current` 在 short-alpha formal 里误继承 `short_alpha` 预算
  - `plain_baseline` 在 dynamic-graph formal 里误继承 `dynamic_graph` 预算
- second-stage frontier 允许只复跑部分家族，但 latest manifest 必须 merge update，不能覆盖未复跑家族。

## 8. recent upgrade gate 规则
- formal 胜出后，不直接静默切换默认执行。
- 必须再过 recent realistic replay gate。
- gate 至少回答：
  - 相对当前 active default 的 full-period 优势
  - 相对当前 active default 的 named-window 胜负
  - 相对同窗 baseline execalign 的优势是否仍在
- 如果要把 fresh production retrain 推进到默认执行，必须在“同一 execution policy”下打赢当前 production root。

## 9. weak-month 诊断与修复规则
- liquid500 主线进入“月度优先”后，先做 weak-month diagnosis，再决定改模型、改 bridge 还是改 execution policy。
- weak-month repair 优先看：
  - weak-month 数量
  - weak-month 所在 regime
  - weak-month 里 candidate 相对 baseline 的月收益差
  - weak-month 里最优 execution policy 相对当前 policy 的 lift
- 如果 weak months 主要集中在少数 regime，优先做 targeted repair，不默认上升到全局 regime-conditioned execution policy。
- 如果扩大的静态 bridge/profile 搜索仍不能打赢当前 formal winner，则停止继续扩大静态集合，转向 targeted weak-month repair。
- targeted weak-month repair 做法默认分两层：
  - 先在 weak months 上学候选 trigger 与候选 policy
  - 再在全部 training months 上给 repair plan 打分，防止靠牺牲强月换取表面弱月修复
- 如果粗 `month_start_regime` trigger 仍伤均值收益，不要回到 broad conditional policy；优先细化到：
  - `month_start_market_state`
  - `regime_market_state`
  - 或其它 month-trigger 级触发键
- 如果 broad conditional policy 与扩大的静态 bridge/profile 搜索都已失败：
  - 冻结新的横向 execution-policy 扫描
  - 除非 active strategy 或 formal protocol 发生变化，否则不重开同类 broad sweep
  - 后续只沿 `month_start_*` / `month-trigger` / `score -> weight -> execution` 修复链继续下钻
- 如果 `regime_signal_shape` 与 `regime_weight_count` 这类 month-start 权重签名 trigger 仍打不赢静态线：
  - 不要继续横向细分更多静态 month-start 类别
  - 把下一步升级为多日 `score / weight` 触发，例如 first-week dispersion、signal persistence、weight concentration drift
- finer trigger review 如果只在旧窗触发、而最新窗保持 `static_only`：
  - 记为 monitored repair candidate
  - 不得直接宣称 active default 已可升级
- 如果某条窄 trigger 只有在 `min_support = 1` 时才扩展到最新窗，但均值收益反而转负：
  - 不得为了“让最新窗也触发”而降低 support 门槛
  - 这类结果应视为 overfit warning，而不是 promotion 依据
- 当前已验证的一条可复用经验是：
  - `regime_market_state` 比粗 `regime` 更适合做窄触发修复
  - 但 support 不够宽时，仍应保持 static fallback，而不是强行全局切换

## 10. 条件化 execution policy 规则
- simple regime-conditioned execution policy 如要进入默认执行，至少先过 leave-window-out formal review。
- 如果 leave-window-out review 对当前静态 policy 为 `0/N` 全败，或 mean excess annual / Sharpe 同时下降，则禁止推到 active default。

## 11. 月度协议规则
- `calendar_months` 是当前主研究时间单位。
- 月度协议作用于：
  - train/valid 切窗
  - train-side eval window
  - adaptive task window
  - monthly summary artifacts
- 研究判读顺序固定为：
  - 先看月度诊断
  - 再看窗口均值
  - 最后看 headline annual / Sharpe
- 月度诊断至少检查：
  - `positive_month_ratio`
  - `median_monthly_return`
  - `worst_monthly_return`
  - `top3_positive_month_share`
  - `longest_negative_streak`

## 12. 架构 refresh 规则
- 重做复杂度 / 深度 / 结构实验时，不再使用旧的固定 `8 epoch` runner 直接下结论。
- 当前正式口径应走：
  - recent 全矩阵
  - recent category winners
  - multi-window formal H2H
  - family epoch budget manifest
  - execution-first + `profit_max_v1`
- 如果某架构分支 formal 仍有 budget pressure，不直接下封死负结论；先标记为“预算仍不足的候选”。
- 如果某架构分支 recent 很强，但 formal 月度稳定性不足，不得直接升级为默认执行候选。
- 如果某架构分支完成预算补齐后仍不能通过 monthly-first / challenger gate，则降级为 monitored branch，不再保持 architecture 主优先级。
- 当前 architecture 线的继续顺序默认改为：
  - 先做 `encoder_transformer_v1` 稳定性修复
  - 再视需要监控 `graph_off_plain`
  - 不再把“更大、更深”本身视为默认升级方向

## 13. execution-side repair 规则
- short-alpha 执行侧后续默认不再横向扫更多 `execution policy / bridge / profile`。
- 如果 month-start 单点 trigger 已证明信息不足，下一步应优先下钻 `first-week / multi-day score trigger`、`first-week / multi-day weight trigger` 与 `score -> weight -> execution` 的多日兑现链。
- `regime_firstweek_combo` 已经形成正证据后：
  - 视为当前已验证的高 ROI repair class，默认优先于继续追加新的静态 `month-start` 分类器
  - 后续应围绕它做收敛和稳健性验证，而不是回到 broad conditional policy
## 14. short-line expert bundle 规则
- `short_expert_monthly_v1` 是当前已验证的 short-line model-side 主候选。
- `short_expert_monthly_v2` 这类“first-week 特征 + state-targeted loss/penalty”大包方案，如果在 `24 -> 32 -> 48` 后变成 budget-stable 仍落后：
  - 记为 monitored negative branch，结论是“这一整包 recipe 没有打赢主候选”，不是“short-line 方向无效”
  - 后续默认拆分成 feature-only / penalty-only ablation
- 不要因为 bundle 更大、更完整，就默认把它当成更高优先级主线。
## 15. checkpoint 选择与续训规则
- `deep_alpha` 主链支持月度 checkpoint objective：
  - `primary_monthly_positive_ratio`
  - `primary_monthly_median_return`
  - `primary_monthly_robust_score`
- short-line expert 这类“月度兑现优先”的候选，默认优先看 `primary_monthly_robust_score`，而不是先回到 `primary_annual_return`
- 如果 output-head experimental branch 没有打赢当前 `ridge` candidate 的 `primary_monthly_robust_score`，即使 annual / Sharpe 或 worst month 略优，也不得覆盖主候选
- 如果要比较不同 checkpoint objective，优先使用：
  - fresh run
  - 或 warm-start restart
- strict resume continuation 必须保持同一 `checkpoint_selection_objective`，不得中途切换。
- 如果 strict resume 因 `feature_names` 等训练链一致性校验失败，必须明确记账并切回 fresh rerun，不得把不一致 continuation 当作同一链证据。
- short-alpha production recipe 的 `24 -> 32 -> 40` extension 是 strict resume continuation，因此保持 `primary_annual_return`。
- 比较不同 checkpoint objective 时，不能只看 candidate 绝对指标是否提高，还要看相对 baseline 的 discrimination 是否变强。
- 如果某 objective 同时抬高 candidate 和 baseline，但 baseline 提升更多，则不得切换主线默认 objective。

## 16. 默认值升级规则
- 只有同一执行口径下同时满足以下条件，才允许升级默认执行：
  - formal rich experiment 胜出
  - recent realistic replay gate 胜出
  - production full-fit promotion 未破坏优势
- 如果 challenger 来自不同股票池主线，必须先补同宇宙 formal；只凭跨股票池 headline 更高，不得进入默认执行升级链。
- 在 production promotion 完成前，研究 winner 只算“候选”，不算“active default”。

## 17. 语言与编码规则
- brain 文档默认使用简体中文。
- shell 运行时输出、进度条文本、运行日志默认使用英文。
- PowerShell 直接读中文 markdown 如果出现乱码，优先用 `yolos` 的 UTF-8 Python 读取，不要误判成文件损坏。

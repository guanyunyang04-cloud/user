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
- execution policy 本身属于研究对象，而不是固定后置适配器。
- 禁止把新研究候选强行塞回旧默认执行壳，再据此宣称“执行端不支持”。
- 如果目标是“净收益最大”，execution policy 的 formal 复核默认优先按 `excess_annual_return` 排序，而不是先用保守 profile 锁死执行层。
- 任何执行升级讨论必须回到：
  - 当前代码
  - 当前特征空间
  - 当前执行桥接
  - 当前现实成本

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
- family epoch budget 的真源固定为：
  - `daily_research/output/deep_alpha_family_epoch_budget_latest.json`
- 当前冻结预算为：
  - `baseline -> 4`
  - `structure -> 12`
  - `short_alpha -> 24`
  - `dynamic_graph -> 16`
- formal runner 默认读取 manifest，不手填统一 `--epochs`。

## 6. frontier 校准规则
- frontier 默认先跑：
  - `4 / 8 / 12 / 16`
- 如果最优点落在最右边界，或右边界仍有 objective-aligned budget pressure，再扩到：
  - `24 / 32`
  - 必要时继续到 `40 / 48 / 64`
- family-default calibration windows 只用于第一阶段。
- dynamic_graph 允许使用更晚 calibration windows，因为天然有效样本起点更晚。

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

## 13. checkpoint 选择与续训规则
- `deep_alpha` 主链支持月度 checkpoint objective：
  - `primary_monthly_positive_ratio`
  - `primary_monthly_median_return`
  - `primary_monthly_robust_score`
- 如果要比较不同 checkpoint objective，优先使用：
  - fresh run
  - 或 warm-start restart
- strict resume continuation 必须保持同一 `checkpoint_selection_objective`，不得中途切换。
- 如果 strict resume 因 `feature_names` 等训练链一致性校验失败，必须明确记账并切回 fresh rerun，不得把不一致 continuation 当作同一链证据。
- short-alpha production recipe 的 `24 -> 32 -> 40` extension 是 strict resume continuation，因此保持 `primary_annual_return`。
- 比较不同 checkpoint objective 时，不能只看 candidate 绝对指标是否提高，还要看相对 baseline 的 discrimination 是否变强。
- 如果某 objective 同时抬高 candidate 和 baseline，但 baseline 提升更多，则不得切换主线默认 objective。

## 14. 默认值升级规则
- 只有同一执行口径下同时满足以下条件，才允许升级默认执行：
  - formal rich experiment 胜出
  - recent realistic replay gate 胜出
  - production full-fit promotion 未破坏优势
- 如果 challenger 来自不同股票池主线，必须先补同宇宙 formal；只凭跨股票池 headline 更高，不得进入默认执行升级链。
- 在 production promotion 完成前，研究 winner 只算“候选”，不算“active default”。

## 15. 语言与编码规则
- brain 文档默认使用简体中文。
- shell 运行时输出、进度条文本、运行日志默认使用英文。
- PowerShell 直接读中文 markdown 如果出现乱码，优先用 `yolos` 的 UTF-8 Python 读取，不要误判成文件损坏。

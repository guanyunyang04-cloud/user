# Daily Research Procedural Memory

## 1. 作用
- 本文件只记录“怎么做”。
- 单次实验过程统一写入 `episodic_memory.md`。
- 稳定事实写入 `semantic_memory.md`。
- 当前判断写入 `working_memory.md`。

## 2. 文档维护规则
- `semantic_memory.md` 只保留长期稳定边界与当前稳定真相。
- `project_map.md` 只保留项目结构、主线地图与决策闭环。
- `working_memory.md` 只保留当前判决、优先级与下一步。
- `action_system.md` 只保留当前高频操作入口。
- 遇到新证据时：
  - 先更新 `working_memory.md`
  - 证据稳定后再沉淀进 `semantic_memory.md`
  - 长过程与原始细节进入 `episodic_memory.md`
- 若 active strategy、execution policy、股票池主线或当前瓶颈发生变化：
  - 同步更新 `working_memory.md`
  - 同步更新 `project_map.md`
  - 必要时同步更新 `semantic_memory.md`
  - 若影响命令入口，再同步更新 `action_system.md`

## 3. 目标函数规则
- 默认目标函数是“执行后净收益最大”，不是“raw holdout 指标最大”。
- 若用户没有明确改目标函数，禁止用“更稳”替代“更赚钱”。
- execution policy 本身属于研究对象，而不是固定后置适配器。
- 禁止把新的研究候选强行塞回旧的默认执行壳，再据此宣称“执行端不支持”。
- 若目标是“净收益最大”，execution policy 的 formal 复核默认优先按 `excess_annual_return` 排序，而不是先用保守 profile 锁死执行层。
- 任何执行升级讨论必须回到：
  - 当前代码
  - 当前特征空间
  - 当前执行桥接
  - 当前现实成本

## 4. formal 判决规则
- formal holdout 负责研究 winner 判决。
- production full-fit 负责默认执行。
- production 结果不得回填为 formal 证据。
- liquid500 默认执行升级判断与 liquid800/mainboard 研究判断必须分开写。
- 不同股票池、不同成本口径、不同 gate 协议下的结果，禁止直接混成单一“全项目最高”结论。
- 若用户要“当前全项目最高净收益”，必须新增一道：
  - global deployable leaderboard
  只有进入同一成本引擎、同一 execution-policy audit 搜索空间后，才允许跨 universe 统一排序。
- 若模型 winner 已稳定、但 execution policy 仍有疑问，可先在同一 raw panel 上做 execution policy audit，再决定是否需要重训模型。

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
- 若最优点落在最右边界，或右边界仍有 objective-aligned budget pressure，再扩到：
  - `24 / 32`
  - 必要时继续到 `40 / 48 / 64`
- family-default calibration windows 只用于第一阶段。
- non-dynamic family 的 second-stage frontier 当前固定使用 `late_preformal`：
  - `20220801_20221031`
  - `20221101_20230131`
  - `20230103_20230131`
- dynamic_graph 允许使用更晚校准窗，因为它天然有效样本起点更晚。

## 7. runner 公平性规则
- specialized formal runner 做家族比较时，必须给每个 profile 解析自己的 native family budget。
- 禁止以下错误：
  - `baseline_current` 在 short-alpha formal 里误继承 `short_alpha` 预算
  - `plain_baseline` 在 dynamic-graph formal 里误继承 `dynamic_graph` 预算
- second-stage frontier 允许只复跑部分家族，但 latest manifest 必须 merge update，不能覆盖未复跑家族。

## 8. recent upgrade gate 规则
- formal 三窗胜出后，不直接静默切换默认执行。
- 必须补 recent realistic replay gate。
- gate 至少要回答：
  - 相对当前 active default 的 full-period 优势
  - 相对当前 active default 的 named-window 胜负
  - 相对同窗 baseline execalign 的优势是否仍在
- 若要把 fresh production retrain 推进到默认执行，必须在“同一 execution policy”下战胜当前 production root；只比较不同 policy 下的单次 uplift 不足以上线。

## 8.1 weak-month 诊断规则
- liquid500 主线已进入“月度优先”阶段后，优先先做 weak-month diagnosis，再决定是否继续改模型、改 bridge、还是改 execution policy。
- weak-month repair 优先盯：
  - 弱月数量
  - 弱月所在 regime
  - 弱月里 candidate 相对 baseline 的月收益差
  - 弱月里最优 execution policy 相对当前 policy 的 lift
- 若弱月主要集中在少数 regime，优先做 targeted repair；不要默认上升到全局 regime-conditioned execution policy。

## 8.2 条件化 execution policy 规则
- simple regime-conditioned execution policy 若要进入默认执行，必须至少通过 leave-window-out formal review。
- 若 leave-window-out review 对当前静态 policy 为 `0/N` 全败，或 mean excess annual / Sharpe 同时下降，则禁止推到 active default。

## 9. TQ 与 raw cache 规则
- 若长矩阵只因 TQ 初始化失败中断：
  - 不重开新 root-tag
  - 复用同一 root-tag
  - 透传 `--force-raw-cache-path`
  - 让 runner 跳过已完成 cells，继续补完剩余 cells
- 当前已验证可复用的 dynamic-graph raw cache：
  - `daily_research/cache/deep_alpha/raw/6e5203c8cdec3a61.pkl`

## 10. 月度协议规则
- `calendar_months` 是当前主研究时间单位。
- 月度协议作用于：
  - train/valid 切窗
  - train-side eval window
  - adaptive task window
  - monthly summary artifacts
- 研究判读顺序固定为：
  - 先看月度诊断
  - 再看窗口均值
  - 最后才看单个 headline annual / Sharpe
- 月度诊断至少检查：
  - `positive_month_ratio`
  - `median_monthly_return`
  - `worst_monthly_return`
  - `top3_positive_month_share`
  - `longest_negative_streak`
- rich experiment summary 必须把 `Monthly Priority Summary` 放在 `Mean Summary` 之前。
- 单个 run 必须优先读取：
  - `primary_research_monthly_summary.csv`
  - `primary_research_monthly_diagnostics.json`
- 若命令已显式给出 `train_end_date + valid_start_date + valid_days`，则显式短窗优先。

## 11. checkpoint 选择与续训规则
- `deep_alpha` 主链现已支持月度 checkpoint objective，至少包括：
  - `primary_monthly_positive_ratio`
  - `primary_monthly_median_return`
  - `primary_monthly_robust_score`
- 若要比较不同 checkpoint objective，优先使用：
  - fresh run
  - 或 warm-start restart
- strict resume continuation 必须保持同一个 `checkpoint_selection_objective`，不得在同一训练链中途切换。
- 现有 short-alpha production recipe 的 `24 -> 32 -> 40` extension 是 strict resume continuation，因此保持 `primary_annual_return`，只把月度 objective 作为下一轮 fresh-run 比较对象。
- 比较不同 checkpoint objective 时，不能只看 candidate 的绝对月度指标是否提高。
- 对 liquid500 主线，正式判决必须至少同时检查：
  - candidate 相对 baseline 的 positive-month ratio delta
  - candidate 相对 baseline 的 median monthly excess delta
  - candidate 相对 baseline 的 mean excess annual / Sharpe delta
  - candidate 对 baseline 的 window wins 是否被削弱
- 若某个 objective 同时抬高了 candidate 和 baseline，但 baseline 提升更大，则不得切换主线默认 objective。

## 12. 默认值升级规则
- 只有同一执行口径下同时满足以下条件，才允许升级默认执行：
  - formal rich experiment 胜出
  - recent realistic replay gate 胜出
  - production full-fit promotion 未破坏优势
- 若 challenger 来自不同股票池主线，必须先补同宇宙 formal；只凭跨股票池 headline 更高，不得进入默认执行升级链。
- 在 production promotion 完成前，研究 winner 只算“候选”，不算“active default”。

## 13. 语言与编码规则
- brain 文档默认使用简体中文。
- shell 运行时输出、进度条文本、运行日志默认使用英文。
- PowerShell 直接读取中文 markdown 若出现乱码，优先用 `yolos` 的 UTF-8 Python 读取，不要误判成文件损坏。

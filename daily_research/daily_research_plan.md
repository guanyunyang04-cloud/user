# Daily Research 研究总纲

## 1. 当前默认决策
当前默认执行主线已经明确并冻结为：

- `advanced_ml (ma50 baseline, lgbm) + liquid500 + next_open`

当前默认口径同时明确为：

- 股票池：
  - 执行端日常使用 `liquid500_latest.txt`
  - 正式研究使用历史滚动 `liquid500 / liquid800`
- 成交假设：
  - `next_open`
- 调仓语义：
  - 日频目标更新，主入口默认 `rebalance_freq=1d`
- 状态门控：
  - `regime_ma_window=50`
  - `regime_max_annual_vol=0.32`
  - `trend_up_low_vol,trend_up_high_vol`
- 模型族：
  - `lgbm`
- 默认训练窗口：
  - `ml_train_window_days=504`

## 2. 当前项目判断
- `advanced_ml` 继续承担执行职责；
- `deep_alpha` 继续承担主研究职责，但尚未晋级执行端；
- `none / v2` 保留为规则层先验，不扩成新的执行主线；
- 连续状态分数保留为诊断层，不进入默认执行软调节逻辑；
- 坏市场专项若重启，主目标固定为“坏市场绝对收益”，当前尚无成熟坏市场专用 profile；
- 坏市场首批专用候选 formal round1 已完成；当前仅 `downdual_reversal_open_hold2 / downdual_reversal_badonly_hold2` 可继续作为近期防守修复候选细化，仍不得按坏市场盈利方案解释或晋级；
- `ma47/48` 左侧边界带、`v21_volume_contraction_015` 等高收益旁支保留在研究附录，不再作为当前执行修复候选。

## 3. 当前优先级
### 优先级 A：完成 `deep_alpha` 当前主线的正式判决
目标：

- 不再扩新分支；
- 把“预训练 + 排序 fine-tune”路线做出可回溯的最终判断。

具体动作：

1. 固定正式框架：
   - 历史滚动高流动性股票池；
   - `next_open`；
   - 多窗口 walk-forward。
2. 优先处理仍然 `undertrained` 的窗口，先解决训练预算问题，再讨论结构升级。
3. 只有在多窗口持续改善且不破坏原本强窗口时，才允许讨论接近执行端。

### 优先级 B：维持执行主线稳态
目标：

- 保持当前默认执行链路可用、可复核、可回滚。

具体动作：

1. 继续按正式研究口径复核 `advanced_ml (ma50 baseline, lgbm)`；
2. 继续使用 `latest_ml_model.json` 的验证摘要与新鲜度保护；
3. 不让研究侧的局部高收益候选静默替换默认值；
4. 若后续再做更长历史复验，优先解决“更早数据边界 + 长训练窗预热”，而不是机械把 `start-date` 往前推。
5. 若重启坏市场专项，统一按“坏市场绝对收益”排序与验收，不再把坏市场超额直接表述为坏市场盈利。

### 优先级 C：把治理规则变成日常流程
目标：

- 避免 README、计划、缓存与输出再次失控堆积。

具体动作：

1. `README.md` 只写当前状态与入口；
2. `daily_research_plan.md` 只写当前决策、优先级和停止规则；
3. `research_log.md` 作为唯一时间顺序实验记录；
4. 定期执行：
   - `python daily_research/tools/doc_guard.py check`
   - `python daily_research/tools/workspace_maintenance.py report`
   - `python daily_research/tools/workspace_maintenance.py archive`

## 4. 当前不推进或已降级的方向
- `ma47/48` 左侧边界带：
  - 当前判断为“季度特定槽位命中”，不再晋级执行端。
- `trend_up_low_vol` 状态专属 ensemble 权重微调：
  - 当前判断为季度集中驱动，不继续扫更大权重网格。
- `v21_volume_contraction_015`：
  - 保留为规则层研究候选，但不正式晋级为默认 `v2.1`。
- 连续状态软调节：
  - 保留为诊断层，不进入当前执行层默认逻辑。
- 状态专属模型研究：
  - 在默认 `lgbm` 已切换且两个真实开仓状态都改善的前提下，暂不重启。

## 5. 已明确的边界
- 早期 `score + 5d` 的结论只严格对应 stage1 基线路径；
- 当前 `advanced_ml` 主线已经正式标准化为“日频目标更新”；
- 当前默认执行主线不是：
  - `advanced_ml + liquid500 + next_open`
- 而是更精确的：
  - `advanced_ml (ma50 baseline, lgbm) + liquid500 + next_open`
- 任何准备接近执行端的新方案，都必须先过：
  - 历史滚动股票池；
  - `next_open`；
  - 多窗口 walk-forward；
  - 强窗口不被破坏。

## 6. 停止规则
以下情况一律不晋级执行端：

1. 只修复最弱窗口，却破坏原本更强窗口；
2. 只在静态最新股票池上好看，未通过历史滚动股票池验证；
3. 只在单次样本或烟测中有效，未通过多窗口 walk-forward；
4. 关键窗口仍显示 `undertrained`；
5. 收益改善主要来自季度集中或少数日期放大；
6. 结论无法在 `research_log.md` 中回溯到对应实验与产物。
7. 所谓“坏市场盈利”若只体现为坏市场超额更高、但坏市场绝对收益仍持续为负，不得按坏市场盈利方案晋级。

## 7. 文档维护规则
- `daily_research/README.md`
  - 只保留当前状态、入口和维护命令。
- `daily_research/daily_research_plan.md`
  - 只保留当前默认决策、优先级和停止规则。
- `daily_research/research_log.md`
  - 只保留时间顺序实验记录与阶段结论。
- 根目录 `README.md`
  - 只负责工作区级治理说明。
- 研究产物归档规则：
  - 以 `daily_research/archive_policy.json` 为准。

## 8. 接下来 6~8 周研究路线图
### 8.1 总原则
- 第一目标不是“追新”，而是确认 `deep_alpha` 是否能稳定超越当前可执行主线；
- 第二目标不是一次性大改，而是把有效升级拆成可复核、可回滚的单模块增量；
- 第三目标才是接入前沿结构，而且必须建立在前两步已经跑通的基础上。

### 8.2 第一阶段：先完成 `deep_alpha` 的最终正式判决
目标：

- 不再横向扩更多支线；
- 把当前 `deep_alpha` 已有能力边界跑清楚；
- 先解决口径与训练预算问题，再讨论新结构。

具体动作：

1. 先做研究口径统一：
   - `deep_alpha` 相关默认值、README 示例与实验元数据统一到“日频目标更新”语义；
   - 正式研究继续固定为：历史滚动流动性股票池 + `next_open` + 多窗口 walk-forward。
2. 做最小充分对照矩阵：
   - encoder 只保留 `gru`、`patch_transformer` 两条主线；
   - pretrain 只比较 `none` vs `masked pretrain`；
   - score head 只比较 `manual`、`ridge`、`lgbm`；
   - ranking 只比较“当前最好配方”与“加入 `ranking/listwise`”。
3. 先消灭关键窗口 `undertrained`：
   - 预训练与 fine-tune 都先扩大 epoch budget；
   - 关键窗口在训练诊断稳定前，不允许下结论说“结构无效”。
4. 这一阶段的晋级门槛：
   - 全样本与最近完整窗口都优于当前最佳 `deep_alpha` 基线；
   - 既有强窗口的超额 Sharpe 不允许明显回撤；
   - 关键窗口不再停留在 `undertrained`；
   - 季度改善不能主要来自少数孤点季度。

### 8.3 第二阶段：优先做成熟、已被验证的结构升级
这阶段只做“文献成熟 + 与现有代码强兼容 + 对收益曲线最可能有帮助”的升级。

优先顺序：

1. `RevIN` 非平稳归一化：
   - 作为可开关输入层接入 `gru / patch_transformer / 后续新编码器`；
   - 重点观察弱窗口、换挡窗口和训练期外漂移窗口是否更稳；
   - 若只改善弱窗口而破坏强窗口，立即停止。
2. `iTransformer` 编码器：
   - 作为 `deep_alpha` 的第三条正式编码器分支；
   - 重点验证更长 lookback 与多变量建模是否优于当前 `patch_transformer`。
3. `PatchTST` 风格增强：
   - 当前 `patch_transformer` 已具备 patch 思路，但还没有显式 channel-independence；
   - 只补“channel-independent patch encoder”这一项，不整套重写。
4. 不确定性校准：
   - 在当前 score/risk 输出上补分位数或区间估计；
   - 用 conformal calibration 做后验校准；
   - 不直接替代主分数，而是先用于 top-k 筛选、仓位打折或风险门附录。

这一阶段的实验顺序固定为：

1. 先在当前最佳 `deep_alpha` 配方上加 `RevIN`；
2. 再引入 `iTransformer`；
3. 再补 `PatchTST` 风格 channel-independence；
4. 最后才做 uncertainty / conformal 接入。

### 8.4 第三阶段：与前沿接轨，但只保留受控试点
以下方向允许跟进，但默认不直接升为主线：

1. `Mamba / selective SSM`：
   - 只作为 `deep_alpha` 的 challenger encoder；
   - 必须放在 `RevIN / iTransformer` 之后；
   - 若样本效率、训练稳定性或窗口一致性不占优，就停止。
2. 双重预训练：
   - 在现有 masked pretrain 之上，增加 contrastive consistency 支线；
   - 只允许做“masked + contrastive”双任务，不再同时叠更多花样。
3. Foundation model 观察线：
   - `Chronos / TimesFM` 只保留为 watchlist；
   - 现阶段不做本地全量微调，不改写主研究框架；
   - 只有在“冻结特征 + 轻量头部”能拿出清晰增量证据时，才考虑升级为正式分支。

### 8.5 具体节奏
- 第 1~2 周：
  - 完成 `deep_alpha` 当前主线正式判决；
  - 顺手统一 `deep_alpha` 的 `rebalance_freq` 与主线元数据语义。
- 第 3~4 周：
  - 做 `RevIN` 与 `iTransformer` 两个成熟升级。
- 第 5~6 周：
  - 做 `PatchTST` 风格 channel-independence 与 uncertainty / conformal 校准。
- 第 7~8 周：
  - 只有前两阶段已经出现稳定增量时，才开启 `Mamba` 或双重预训练试点。

### 8.6 新技术的统一验收标准
以下条件必须同时满足，才允许该方向继续投入：

1. 全样本与最近完整窗口都改善；
2. 既有强窗口不被明显破坏；
3. 多季度改善具有连续性，而不是单季度脉冲；
4. `turnover / drawdown / 持仓集中度` 不出现不可解释恶化；
5. 关键窗口训练诊断不再长期显示 `undertrained`；
6. 能形成独立产物、独立摘要和可回溯日志。

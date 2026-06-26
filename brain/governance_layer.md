# 主脑治理层

快照日期：`2026-06-27`

## 1. 治理目标
治理层不是审批流程。它负责让受保护对象的不变量清楚、可触发、可验证，并避免脑区退化成全局禁令清单。

默认闭环：

1. `Understand`：理解用户目标。
2. `Identify Objects`：识别任务触碰的对象和方法。
3. `Activate Boundaries`：只激活相关对象的保护语义。
4. `Act`：直接推进任务。
5. `Validate`：用足够支撑当前结论的检查验证。
6. `Write Back`：把当前状态、长期事实或长证据写回对应对象。

## 2. 保护对象类型

### `canonical_or_unique_data`
- 包括 QDP canonical lake、registry pointer、policy bundle、memmap、唯一研究证据和不可重建资产。
- 激活方法：rebuild、switch_pointer、cleanup、delete、migrate。
- 不变量：有替代指针、dry-run 或抽样验证前，不把唯一资产当普通缓存处理。

### `pit_or_label_semantics`
- 包括 PIT/no-leakage、label completeness、future availability 和 evidence grade。
- 激活方法：数据集构建、特征/标签修改、训练、评估、模型结论。
- 不变量：未观测标签、未来信息、低预算实验和 incomplete run 不写成正式结论。

### `active_execution_artifact`
- 包括 live/default/paper/broker、active artifact、promotion gate 和交易计划。
- 激活方法：activate、restore、update、generate_trade_plan、paper/live/broker wiring。
- 不变量：非执行任务不触碰；执行相关修改需要显式授权和分脑 promotion 边界。

### `secret_or_external_state`
- 包括密钥、账号、外部服务状态、远端部署和真实交易服务。
- 激活方法：create、rotate、write、deploy、connect。
- 不变量：不静默改变，不写入 brain，不在普通文档中暴露。

### `cross_project_dirty_work`
- 包括其它项目未归属 dirty paths、后台进程、端口、GPU、数据 provider 任务和输出。
- 激活方法：manage、delete、reuse、commit、wait。
- 不变量：无归属或授权时只报告存在，不下钻管理，不混提交。

## 3. 经验对象
- `workspace_brain_skill`：入口提示器，不是审批者。
- `route / capsule / bootstrap`：传感器，不替代 agent 判断。
- `doc_guard / integrity_check / health`：验证方法，不是每次任务的默认仪式。
- `Agent Meta Protocol`：agent 自我修正与经验写回对象；它记录学习机会和授权状态，不替代用户目标或对象证据。
- `proposal`：用于不确定或高风险的脑区演化；用户已确认的低风险语义整理可直接实施。
- `compatibility`：只有真实调用证据、不可替代证据价值或外部接口责任时保留。

## 4. 写回纪律
- 当前状态写回 state。
- 稳定定义和对象语义写回 knowledge。
- 对象模型和结构边界写回 architecture。
- 方法入口和验证方式写回 operations。
- 保护对象与不变量写回 governance。
- 项目事实写回相关分脑。
- 长证据写入 references。

## 5. 脑区减负
- 热路径文件只保留当前对象、当前状态、当前方法和当前边界。
- 长历史、完整命令、dated review、旧事故和实验细节进入 references。
- 当规则冲突、文档读取成本超过任务收益、兼容入口造成歧义或测试锁住旧设计时，优先收敛对象定义，而不是继续叠加规则。
- 审计入口：`C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py brain-burden-audit --cwd . --mode compact`

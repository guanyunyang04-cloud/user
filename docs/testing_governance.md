# 测试治理与减负策略

## 目标

工作区测试默认服务于快速、安全的改动闭环，而不是证明所有研究结论。测试体系按“轻默认、硬关键、慢验证可调度”治理：普通任务只跑改动面必需检查，研究、训练、全量数据和外部服务验证进入显式车道。

## 测试车道

| 车道 | 目标预算 | 默认性 | 用途 |
|---|---:|---|---|
| `smoke` | 2 分钟内 | 默认 | `selective_verification` 推荐的 changed-surface 阻塞命令 |
| `project` | 8 分钟内 | 普通收口 | 项目普通测试，排除 slow/research/external/benchmark |
| `full` | 不设固定预算 | 阶段收口 | 全部确定性项目测试，可包含 deferred deterministic suite |
| `research` | 不设固定预算 | 明确研究任务 | 训练、回测、模型、memmap 构建、研究矩阵 |
| `external` | 不设固定预算 | 手动或定时 | 网络、真实 provider、第三方服务、benchmark |

默认入口：

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.selective_verification --paths <changed_paths> --json
```

普通改动执行 `blocking_commands`，并查看 `lane_commands` 和 `test_strategy` 判断是否需要追加 `project/full/research/external`。

## 减负原则

- 默认测试保护行为契约、安全边界和数据 schema，不复刻完整生产流程。
- 单测优先使用 synthetic/minimal fixture；真实大数据、真实 memmap、真实 provider 只在专项车道跑。
- 大 memmap 构建、模型训练、全量回测、BaoStock/TDX/live provider probe 不进入默认 `blocking_commands`。
- 研究结论进入 report、reference 或 evidence registry；测试只保护报告生成器、schema、gate 和 no-leakage 规则。
- 旧机制删除或收敛后，对应旧测试同步归档或删除；不让测试冻结旧设计。
- 多个测试验证同一契约时，合并成一个清楚的 contract test。

## 项目口径

- `quant_data_platform`：测试少且保护 canonical 数据基底，默认基本全跑；full sharded memmap 构建仍只作专项验证。
- `daily_research`：训练、全量 memmap、研究矩阵、长回测归 `research/slow`；默认只保留 profile、registry、manifest、PIT/no-leakage、active 边界和 changed-surface 契约。
- `traditional_quant_research`：多数实验回归归 `research`；普通改动跑同面测试，整包测试只在阶段收口或候选 gate 变更时跑。
- `daily_stock_analysis-main`：API/config/pure service contract 可进普通车道；LLM、搜索、网络、通知、行情 provider、benchmark 归 `external` 或 `benchmark`。
- `tools/brain`：保留路由、project commit、doc guard、选择性验证、项目边界等治理测试；长审计和全量 health 只作维护车道。

## 保留硬边界

不得为减负削弱以下测试或守卫：

- canonical registry、bundle、memmap 指针和 signature 防误命中。
- 数据清理 dry-run、路径边界和不可恢复删除保护。
- PIT、严格滞后、未来函数、OOS/role-year 边界。
- `daily_research/output/active_execution_strategy.json` blocker。
- 主脑路由、项目命名空间、自动提交闭环和 doc guard。
- live/default/execution promotion 边界。

## 瘦身流程

对每个测试文件打一个处置标签：

- `keep`：关键契约或安全边界。
- `compress`：与其它测试重复，合并为更小契约。
- `move_slow`：标记 slow/research/data_heavy/external，不进默认。
- `archive`：历史研究证据，迁入 report/reference。
- `delete`：只保护已废弃机制且无证据价值。

执行顺序：

1. 先运行 `selective_verification` 获取当前改动面。
2. 对慢测试先移动车道或拆出 nodeid，不急于删除。
3. 对旧机制测试，先确认对应代码路径无真实调用证据。
4. 每次清理只提交当前项目或 workspace 允许范围内的测试治理改动。
5. 收口时至少跑本轮 `blocking_commands`、`git diff --check`、必要 brain/doc guard。


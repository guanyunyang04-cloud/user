# T0 Project Action System

快照日期：`2026-04-12`

## 1. 作用
本文件保存 `t0_project` 的执行抽象设计与运行边界。

## 1.1 接管快路
- 当前默认接管入口先看：
  - `t0_project/brain/identity_layer.md`
  - `t0_project/brain/handoff_packet.md`
  - `t0_project/brain/temporal_state.md`

### Bootstrap
```powershell
python daily_research/tools/brain_bootstrap.py --child t0_project --json
```

## 2. 目标分层
系统拆成四层：

1. 信号层
2. 风控层
3. 委托层
4. 回报层

## 3. 当前文件
- `execution/models.py`
  - 统一数据结构：信号、订单、持仓、账户、风控结果
- `execution/broker.py`
  - 交易接口抽象层
- `execution/risk.py`
  - 基础交易风控
- `execution/order_manager.py`
  - 信号到订单的编排与送审
- `execution/paper_broker.py`
  - 模拟券商
- `execution/live_broker.py`
  - 真实交易适配器骨架
- `execution/config.py`
  - 配置结构与加载函数
- `execution/factory.py`
  - 根据 `paper/live` 构建执行管理器

## 4. 运行模式
- `paper`
  - 默认模式，用于验证执行流程
- `live`
  - 当前不会真的下单，仍是保护性骨架

## 5. 接入顺序
1. 先接 `PaperBroker`
2. 再接人工确认下单
3. 最后才接真实自动交易接口

## 6. RL 执行层路线入口
### 6.1 数据采集
```bash
python t0_project/rl_agent/data_collector.py
```

### 6.2 训练入口
```bash
python t0_project/rl_agent/train_ppo_agent.py --data-path t0_project/rl_agent/data/600536.SH_1m_history.csv --model-name recurrent_ppo_exec_r1 --vecnormalize-name vecnormalize_exec_r1.pkl --total-timesteps 200000
```

### 6.3 回放与调试入口
```bash
python t0_project/rl_agent/debug_inference.py --data-path t0_project/rl_agent/data/600536.SH_1m_history.csv --model-path t0_project/models/recurrent_ppo_exec_r1.zip --vecnormalize-path t0_project/models/vecnormalize_exec_r1.pkl
```

### 6.4 当前约束
- RL 当前只在 `paper` / 回放链路里验证执行质量。
- 当前目标是：
  - timing
  - sizing
  - inventory 管理
- 当前不把 RL 结果直接外推到 `daily_research` 的日线 alpha 主线。

## 7. 当前安全边界
- `live` 当前不会真的下单
- 若信号进入 `LiveBrokerAdapter`，未实现能力会抛出 `NotImplementedError`
- 这属于保护，不是 bug

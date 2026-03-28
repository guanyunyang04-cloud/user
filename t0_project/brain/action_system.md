# T0 Project Action System

## 1. 作用
本文件保存 `t0_project` 的执行抽象设计与运行边界。

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

## 6. 当前安全边界
- `live` 当前不会真的下单
- 若信号进入 `LiveBrokerAdapter`，未实现能力会抛出 `NotImplementedError`
- 这属于保护，不是 bug

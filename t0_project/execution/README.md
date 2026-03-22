# 执行层设计

这部分不是直接下单代码，而是给当前 `integrated_tq_strategy.py` 预留一层可扩展的执行架构。

目标是把系统拆成四层：

1. 信号层
- 日线选股
- 盘中抢筹建仓
- 超跌低吸
- 轨道高抛
- 强制止损

2. 风控层
- 限制单笔委托金额
- 限制单票仓位
- 限制单日交易次数
- 限制最小交易单位
- 在真实接券商前，先把交易规则固化

3. 委托层
- 接收信号
- 生成标准化订单对象
- 调风控审批
- 把通过的订单发送给交易接口
- 维护订单状态

4. 回报层
- 回写成交均价
- 更新持仓
- 更新现金
- 识别拒单、撤单、部分成交

## 当前文件

- `execution/models.py`
定义统一数据结构：信号、订单、持仓、账户、风控结果

- `execution/broker.py`
定义交易接口抽象层，后续接券商或通达信交易模块都从这里接

- `execution/risk.py`
定义基础交易风控

- `execution/order_manager.py`
把信号转换成订单并送审、发送

- `execution/paper_broker.py`
一个内存版模拟券商，主要用于先把执行流程跑通

- `execution/live_broker.py`
真实交易适配器骨架。当前故意不实现真实下单，只保留接口边界

- `execution/config.py`
真实交易配置结构与加载函数

- `execution/factory.py`
根据 `paper/live` 构建执行管理器

## 如何映射现有信号

建议映射如下：

- `抢筹建仓` -> `SignalType.BUY_ENTRY`
- `超跌低吸` -> `SignalType.BUY_ADD`
- `急拉抢筹` -> `SignalType.BUY_ADD`
- `急跌出逃` -> `SignalType.SELL_REDUCE`
- `轨道高抛` -> `SignalType.SELL_REDUCE`
- `强制熔断` -> `SignalType.SELL_EXIT`

## 运行模式

主程序现在支持：

- `--execution-mode paper`
默认，使用 `PaperBroker` 模拟执行

- `--execution-mode live --broker-config path/to/config.json`
切换到真实交易适配器骨架

注意：
- `live` 当前不会真的下单
- 一旦有买卖信号，会进入 `LiveBrokerAdapter`
- 由于下单、撤单、持仓、账户查询尚未绑定真实 API，会抛出 `NotImplementedError`
- 这是刻意的保护，不是 bug

## 配置样例

参考：
- `execution/live_broker_config.example.json`

建议把密码放在环境变量中，例如：

```powershell
$env:T0_BROKER_PASSWORD="your_password"
```

## 接入顺序

建议分三步：

1. 先接 `PaperBroker`
- 不碰真实委托
- 验证状态机和风控

2. 再接“人工确认下单”
- 信号出来后生成标准化订单建议
- 人工确认后再实际发送

3. 最后才接真实自动交易接口
- 需要账户、委托、成交、撤单、风控、异常恢复全部打通

## 真实适配器需要补的最小能力

`LiveBrokerAdapter` 至少需要补全这些方法：

- `place_order`
- `cancel_order`
- `get_order`
- `list_open_orders`
- `get_positions`
- `get_account`

如果缺其中任意一块，都不建议实盘自动交易。

## 为什么不直接改成自动下单

因为当前程序还缺这些关键能力：

- 订单状态持久化
- 成交回报回写
- 重复下单保护
- 撤单与超时重试
- 断线恢复
- 实盘资金/持仓同步

没有这些，直接自动下单风险太高。

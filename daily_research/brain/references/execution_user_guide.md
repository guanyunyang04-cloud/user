# Daily Research 执行控制台使用教程

权威操作真源：`daily_research/brain/operations_center.md`。本教程只保留面向使用者的手动操作说明。

## 1. 使用前提

- 所有程序都必须在 `yolos` 环境下运行。
- 依赖环境真源是 `daily_research/environment.yml`。
- 如果本机环境刚调整过，先同步环境：

```powershell
conda env update -f daily_research/environment.yml --prune
```

## 2. 启动 Web

推荐直接使用独立入口：

```powershell
conda run -n yolos python daily_research/execution/run_execution_web.py --port 8765
```

也可以通过统一应用入口启动：

```powershell
conda run -n yolos python daily_research/execution/run_execution_app.py web --port 8765
```

启动后打开：

```text
http://127.0.0.1:8765
```

说明：

- 控制台默认只监听 `127.0.0.1`，不要改成公网地址对外暴露。
- 如果 React build 不存在，页面会返回 `404`，不再回退到 Jinja 模板页。
- 执行端不提供任何后台定时或一键每日流水线；任务只由页面按钮或明确 CLI 单项命令触发。

## 3. 每日手动流程

在 Web 的“帮助”页按顺序操作：

1. `刷新数据/信号`
2. `生成交易计划`
3. `模拟账户过账`
4. `复核状态`

如果某一步返回 blocked、failed 或明显 stale，先进入对应页面查看证据路径和作业日志，不要继续把后续步骤当作今日完成。

## 4. 页面说明

### 总览

- 查看最新 daily verdict、目标交易日、阻断原因、交易计划、模拟账户和最近作业。
- daily verdict 是只读事实参考；今日是否可用还要结合帮助页流程和作业证据。

### 数据

- 查看 data platform refresh、lake dataset、provider health、production signal panel 和 readiness 阻断。
- 数据缺口严格阻断：候选日 `market_daily` 为空或 required domain blocked 时，不生成新计划。

### 交易计划

- 查看结构化动作表、持仓、watchlist、模型信息和 TXT 原文。
- 只有手动确认数据/信号已就绪后才生成交易计划。

### 模拟账户

- 查看现金、持仓、pending orders、成交流水和区间收益。
- `模拟账户过账` 只按可用价格处理；缺执行日 open、停牌、涨跌停或现金不足时订单保持 pending/blocked。

### 作业

- 查看任务历史、runner/business/artifact 状态、stdout/stderr 和证据路径。
- 失败或阻塞作业可在确认原因后恢复；不要在旧作业仍运行时强制解锁。

### 系统

- 查看 doctor、锁状态、latest daily verdict 和关键证据路径。
- 系统页不展示完整 Runtime JSON；旧 runtime 只作为归档证据。

## 5. 常见操作

### 检查数据 readiness

- 页面路径：`数据`
- API：`GET /api/data-readiness`

### 刷新数据或信号

- 页面路径：`帮助` 或 `数据`
- 任务名：`data-platform-refresh` 或 `refresh-production-live-panels`
- 常见做法：使用默认 as-of 日期、股票池和 domains；必要时在数据页显式填写参数。

### 生成默认交易计划

- 页面路径：`帮助` 或 `交易计划`
- 任务名：`trade-plan`
- 默认候选 profile：`active_execution_strategy`

### 模拟账户过账

- 页面路径：`帮助` 或 `模拟账户`
- 按最新交易计划把可成交订单写入本地 paper ledger。

### 查看最近执行历史

- 页面路径：`作业`
- 重点看业务状态、runner 状态、artifact 状态、开始/完成时间和证据路径。

### 清理失效锁

- 页面路径：`系统`
- 只在确认旧作业已经不再运行时使用。

## 6. 排障建议

### 页面打不开

- 先确认启动命令是否用的是 `yolos`。
- 再确认端口是否被占用。

### 体检不是正常

- 优先修环境、manifest、持仓文件或依赖缺失。
- 不要带着异常状态直接重跑任务。

### 数据不可用

- 查看 `GET /api/data-readiness` 和数据页 provider 证据。
- 如果目标日行情为空，只能等待或人工补数入湖；不得用上一完整交易日伪装今日完成。

### 作业成功但结果不对

- 不要只看退出码。
- 继续检查业务状态、artifact 状态、`latest_trade_plan.txt`、manifest 和输出目录。

### 模拟订单没成交

- 检查 pending order 的 `reason`。
- 常见原因包括 `missing_execution_open`、停牌、涨跌停、现金不足或订单执行日仍在未来。

## 7. 高风险提醒

- 不要把控制台改成公网监听。
- 不要绕过 `yolos` 直接使用当前 shell Python。
- 不要在旧作业仍存活时强制解锁。
- 不要把旧 runtime 当成新事实来源。
- 不要在目标日行情为空时用上一完整交易日伪装今日完成。
- 不要把 production 重训、active manifest 写入或 promotion 当成默认动作；这些操作必须显式提交并确认。

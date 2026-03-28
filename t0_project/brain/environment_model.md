# T0 Project Environment Model

## 1. 根环境
- 工作区根目录：`H:\new_tdx64\PYPlugins\user`
- 默认 Shell：`PowerShell`

## 2. 常用命令
语法编译检查：

```bash
python -m compileall t0_project
```

只清理 `t0_project` 相关生成物：

```bash
python daily_research/tools/workspace_maintenance.py clean --targets pycache,tensorboard,t0_backtest_output,t0_logs
```

查看整个工作区体检：

```bash
python daily_research/tools/workspace_maintenance.py report
```

## 3. 当前边界
- 本分脑只记录 `t0_project` 自己的运行与维护口径
- 跨项目治理以上级主脑为准

# 运行环境基线

快照时间：`2026-04-11`  
时区：`Asia/Shanghai`  
工作区根目录：`H:\new_tdx64\PYPlugins\user`  
默认 Shell：`PowerShell`

## 1. 默认运行口径
- `daily_research` 的正式研究、训练、回测、执行，默认使用 `yolos`。
- 推荐解释器固定为：
  - `C:\Users\ASUS\miniconda3\envs\yolos\python.exe`
- 不依赖“当前 shell 已激活 conda 环境”的隐式状态。
- 对外写命令时，优先写解释器绝对路径。
- brain 文档默认使用简体中文；终端运行时输出默认使用英文。

## 2. Python 环境
- `base`
  - 路径：`C:\Users\ASUS\miniconda3\python.exe`
  - 用途：轻量维护脚本
  - 限制：不作为 `daily_research` 正式研究解释器
- `yolos`
  - 路径：`C:\Users\ASUS\miniconda3\envs\yolos\python.exe`
  - 用途：正式研究、训练、回测、执行默认解释器

## 3. 当前依赖基线
- `pandas==2.3.2`
- `numpy==2.2.6`
- `torch==2.5.1+cu121`
- `lightgbm==4.6.0`
- `CUDA available=True`

## 4. 推荐调用方式
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\execution\run_trade_plan.py
```

## 5. 编码与读取提示
- PowerShell 直接读取中文 markdown 时可能出现显示乱码。
- 需要精确检查中文 brain 文档时，优先使用：
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" -c "from pathlib import Path; print(Path(r'H:\new_tdx64\PYPlugins\user\daily_research\brain\working_memory.md').read_text(encoding='utf-8'))"
```

## 6. 缓存与产物热区
- 研究缓存根目录：
  - `daily_research/cache/deep_alpha`
- 研究产物根目录：
  - `daily_research/output`
- dynamic-graph 当前已验证可复用 raw cache：
  - `daily_research/cache/deep_alpha/raw/6e5203c8cdec3a61.pkl`

## 7. 工作区维护入口
```powershell
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\tools\workspace_maintenance.py report
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\tools\workspace_maintenance.py archive --limit 20
& "C:\Users\ASUS\miniconda3\envs\yolos\python.exe" daily_research\tools\workspace_maintenance.py clean --targets pycache
```

## 8. 适用范围
- 本文件只维护运行环境与工具基线。
- 研究判决不写入本文件。

## 9. 当前机器纪律补充
- 本机正式训练、formal 回放、recent 回放继续锁定 `yolos`。
- 本机正式训练默认前台执行，不再默认使用后台监控器或后台训练。
- 本机正式训练默认 `num_workers = 0`。
- 本机正式训练默认 `pin_memory = false`。
- 未经用户明确批准，不重新启用 CPU 并行供数。

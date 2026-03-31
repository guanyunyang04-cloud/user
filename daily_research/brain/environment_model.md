# 运行环境基线

本文档记录 `daily_research` 分脑当前默认使用的运行环境，作为后续研究、执行、维护与 AI 协同时的统一调用口径。

快照时间：`2026-03-31`  
时区：`Asia/Shanghai`  
工作区根目录：`H:\new_tdx64\PYPlugins\user`  
默认 Shell：`PowerShell`

## 1. 当前推荐口径
- `daily_research` 的正式研究、`deep_alpha` 训练/回测、最小充分对照矩阵，默认使用 `quant` 环境。
- 不再默认使用 `base` 环境直接运行研究脚本，因为该环境缺少核心研究依赖。
- 对外记录命令时，优先给出可直接执行的解释器绝对路径，避免会话切换后口径漂移。
- 跨项目治理与上级脑入口见：
  - `brain/master_brain.md`

## 2. Python 解释器
### 默认 `python`
- 路径：`C:\Users\ASUS\miniconda3\python.exe`
- 版本：`3.12.4`
- 状态：可用于轻量维护脚本，但当前缺少 `pandas`

### 推荐 `quant` Python
- 路径：`C:\Users\ASUS\miniconda3\envs\quant\python.exe`
- 版本：`3.9.25`
- 状态：当前 `daily_research` 正式研究默认解释器

## 3. `quant` 关键依赖基线
- `pandas==2.3.3`
- `numpy==2.0.2`
- `torch==2.8.0+cpu`

## 4. 可用 conda 环境
- `base`
- `jieshun`
- `labelImg37`
- `quant`
- `yolos`

## 5. 推荐调用方式
### 方式 A：直接指定解释器
```powershell
& "C:\Users\ASUS\miniconda3\envs\quant\python.exe" daily_research\deep_alpha\run_minimal_matrix.py --phase backbone --root-tag deep_alpha_minimal_matrix_round1
```

### 方式 B：先激活环境，再运行
```powershell
conda activate quant
python daily_research/deep_alpha/run_deep_alpha_research.py --help
```

## 6. AI 协同工具入口
### Gemini 模块状态
```powershell
daily_research\tools\gemini_frontend.cmd status
daily_research\tools\gemini_frontend.cmd close
```

### 当前协同边界
- 从 `2026-03-29` 起，整个 Gemini 协作模块暂时中止。
- `gemini_frontend` 只保留停用占位和清理入口，不再作为默认研究、复核或收尾工具。
- 关闭或查询状态可以继续走：
  - `daily_research\tools\gemini_frontend.cmd status`
  - `daily_research\tools\gemini_frontend.cmd close`
- `ask / closeout / doctor / pin / unpin / sessions / open` 当前均视为停用动作。

## 7. 工作区维护入口
- 工作区维护与归档工具：
  - `python daily_research/tools/workspace_maintenance.py report`
  - `python daily_research/tools/workspace_maintenance.py archive --limit 20`
  - `python daily_research/tools/workspace_maintenance.py clean --targets pycache`
- 当 `daily_research/cache` 或 `daily_research/output` 出现热区告警时：
  - 默认先看 `report`
  - 再做 `archive` 预演
  - 不直接删除当前活跃实验目录、最新执行产物或仍在对照中的 artifact

## 8. 当前适用范围
- 默认使用 `quant` 的脚本：
  - `daily_research/deep_alpha/pretrain_deep_alpha_encoder.py`
  - `daily_research/deep_alpha/run_deep_alpha_research.py`
  - `daily_research/deep_alpha/run_minimal_matrix.py`
  - 其他依赖 `pandas / numpy / torch` 的正式研究脚本
- 可继续用轻量环境维护的工具：
  - `daily_research/tools/doc_guard.py`
  - `daily_research/tools/workspace_maintenance.py`
  - `daily_research/tools/gemini_frontend.ps1`
  - 其他不依赖研究栈的文档、清理和协同工具

## 9. 使用注意
- `run_minimal_matrix.py` 会继承 `sys.executable` 写出阶段命令清单，因此必须从 `quant` 环境启动，才能让后续命令文件保持正确解释器路径。
- 当前 Windows / PowerShell 环境下，`conda run -n quant ...` 在中文进度输出较多时可能触发 `gbk` 回显异常；
  对正式研究脚本，优先直接使用 `C:\Users\ASUS\miniconda3\envs\quant\python.exe`，避免“脚本已跑完但 `conda run` 在打印输出时失败”的假异常。
- 如果未来升级了解释器、切换了核心包版本，或新增 GPU / CUDA 依赖，应先更新本文档，再启动新的正式实验批次。
- 如果未来重新启用 Gemini 协作模块，应先更新 `procedural_memory.md`，再恢复 `brain_manifest.json` 中的默认口径。

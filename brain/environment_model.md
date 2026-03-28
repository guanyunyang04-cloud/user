# Main Environment Model

## 1. 根环境
- 工作区根目录：`H:\new_tdx64\PYPlugins\user`
- 默认 Shell：`PowerShell`
- 当前日期：`2026-03-28`
- 时区：`Asia/Shanghai`

## 2. 共享工具
- 文档守卫：
  - `python daily_research/tools/doc_guard.py check`
- 工作区体检：
  - `python daily_research/tools/workspace_maintenance.py report`
- Gemini 前台协同：
  - `daily_research\tools\gemini_frontend.cmd open`
  - `daily_research\tools\gemini_frontend.cmd status`
  - `daily_research\tools\gemini_frontend.cmd close`
  - `daily_research\tools\gemini_frontend.cmd sessions`

## 3. 解释器口径
- 轻量维护默认：
  - `C:\Users\ASUS\miniconda3\python.exe`
- 正式研究默认：
  - `C:\Users\ASUS\miniconda3\envs\quant\python.exe`

## 4. 根级维护原则
- 根级只维护脑结构、工具链和跨项目治理
- 项目内部具体环境细节写进各自分脑的 `environment_model.md`

# Main Environment Model

## 1. 根环境
- 工作区根目录：`H:\new_tdx64\PYPlugins\user`
- 默认 Shell：`PowerShell`
- 当前日期：`2026-03-28`
- 时区：`Asia/Shanghai`

## 2. 共享解释器
- 轻量维护默认：
  - `C:\Users\ASUS\miniconda3\python.exe`
- 正式研究默认：
  - `C:\Users\ASUS\miniconda3\envs\quant\python.exe`

## 3. 共享工具
- 脑网络守卫：
  - `python daily_research/tools/doc_guard.py check`
- 工作区体检：
  - `python daily_research/tools/workspace_maintenance.py report`
- Gemini 常驻协同：
  - `daily_research\tools\gemini_frontend.cmd open`
  - `daily_research\tools\gemini_frontend.cmd status`
  - `daily_research\tools\gemini_frontend.cmd close`
  - `daily_research\tools\gemini_frontend.cmd sessions`
  - `daily_research\tools\gemini_frontend.cmd ask -Prompt "..."`
- 当前默认规则：
  - Gemini 只在前台常驻模式下调用
  - 用户关掉前台窗口，即视为本轮 Gemini 协作模式结束

## 4. 编码与读取口径
- 脑文件统一使用 UTF-8
- Windows 下读取脑文件时，优先显式使用：
  - `Get-Content -Encoding UTF8 <path>`

## 5. 根级维护原则
- 根级只维护脑结构、共享工具链和跨项目治理
- 项目内部环境细节写进各自分脑的 `environment_model.md`

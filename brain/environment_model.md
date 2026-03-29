# Main Environment Model

## 1. 根环境
- 工作区根目录：`H:\new_tdx64\PYPlugins\user`
- 默认 Shell：`PowerShell`
- 当前日期：`2026-03-29`
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
- Gemini 标准协同：
  - `daily_research\tools\gemini_frontend.cmd ask -Prompt "..."`
  - `daily_research\tools\gemini_frontend.cmd ask -Prompt "..." -FreshSession`
  - `daily_research\tools\gemini_frontend.cmd ask -Prompt "..." -Escalate`
  - `daily_research\tools\gemini_frontend.cmd closeout -WorkSummary "..." -NextStep "..."`
  - `daily_research\tools\gemini_frontend.cmd sessions`
  - 可选人工前台：`open / status / close`
  - 升级隔离前台：`open -ForceNew -Escalate`
- 当前默认规则：
  - Gemini 默认走后台 `--resume` 续接
  - 前台窗口只作为可选人工交互入口
  - 若出现明显幻觉/陈旧上下文，先 `-FreshSession`，再视风险升级到 `-Escalate`

## 4. 编码与读取口径
- 脑文件统一使用 UTF-8
- Windows 下读取脑文件时，优先显式使用：
  - `Get-Content -Encoding UTF8 <path>`

## 5. 根级维护原则
- 根级只维护脑结构、共享工具链和跨项目治理
- 项目内部环境细节写进各自分脑的 `environment_model.md`

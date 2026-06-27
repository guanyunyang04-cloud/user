---
applyTo: "apps/dsa-web/**,apps/dsa-desktop/**,scripts/run-desktop.ps1,scripts/build-desktop*.ps1,scripts/build-*.sh,docs/desktop-package.md"
---

# Client Objects

`web_client`: Vite + React 前端，沿用当前 API、路由、状态管理、Markdown/图表渲染和认证模式。

`desktop_client`: Electron 桌面端，依赖 Web 构建产物、本地后端启动链路和桌面打包脚本。

`client_api_contract`: API 字段、认证状态、路由行为、报告 payload、轮询状态和本地后端启动方式会同时影响 Web 与 Desktop。

`validation_method`: Web 改动优先运行 `cd apps/dsa-web && npm ci && npm run lint && npm run build`；桌面改动在可行时先构建 Web，再构建 Electron。

`platform_limit`: 如果平台限制阻止完整 Electron 验证，交付说明记录已验证层级和剩余风险。

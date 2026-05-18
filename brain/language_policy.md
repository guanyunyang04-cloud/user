# 脑区语言规范

快照日期：`2026-05-18`

## 默认策略
- 脑区采用 `zh_semantic_en_identifiers_v1`：中文语义 + 英文工程标识。
- 状态判断、研究结论、治理规则、用户可读解释优先使用中文。
- CLI、JSON key、workflow id、dataset id、model family、tag、路径、函数名保留英文。
- 不把正常英文工程标识强行翻译，避免破坏工具、搜索和跨 agent 兼容性。
- 中文正文不使用 PowerShell 默认编码写入；使用 `apply_patch` 或显式 UTF-8 工具链。

## 检查口径
- 核心 Markdown 脑区文档应包含中文语义。
- JSON、命令、路径和代码标识符允许英文。
- 真实 mojibake 或 replacement char 属于文档损坏；终端显示乱码不等于文件损坏，需用 UTF-8 读取确认。

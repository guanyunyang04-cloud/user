# Daily Stock Analysis 知识中枢

## 1. 稳定事实
- `daily_stock_analysis-main` 保存多市场 AI 股票分析产品的稳定认知
- 它服务于 `src / api / apps / bot / data_provider / tests`
- 产品定位来自公开 README 的稳定内容：
  - 面向 A 股、港股、美股与美股指数的 AI 自选股分析系统
  - 输出决策仪表盘、买卖点位、操作检查清单、大盘复盘与多渠道推送
  - Web、Bot、API、桌面应用属于同一产品线的不同入口
- 核心能力：
  - 技术面、筹码分布、舆情、实时行情、基本面聚合与市场策略系统
  - 智能导入图片、CSV/Excel、剪贴板，并通过 Vision LLM 与本地/拼音/数据源解析代码
  - 历史记录、回测验证、持仓管理、Agent 策略问股与内置策略库
- 数据与模型生态：
  - 模型侧支持 AIHubMix、Gemini、OpenAI 兼容、DeepSeek、通义千问、Claude、Ollama 等，统一经 LiteLLM 风格配置
  - 行情侧支持 AkShare、Tushare、Pytdx、Baostock、YFinance
  - 新闻与搜索侧支持 Tavily、SerpAPI、Bocha、Brave、MiniMax、SearXNG 等
  - 通知侧支持企业微信、飞书、Telegram、Discord、Slack、钉钉、邮件、Pushover、PushPlus、Server 酱和自定义 Webhook
- 基本面聚合采用 fail-open 降级：第三方能力失败或超时时不得阻断主分析链路
- 公开 README 可保留用户安装、配置、演示与免责声明，但本产品分脑的 AI 接管、结构边界和稳定规则必须先沉淀在 `daily_stock_analysis-main/brain/`
- `AGENTS.md`、`CLAUDE.md`、`.github/copilot-instructions.md`、`.github/instructions/*.instructions.md`、`SKILL.md` 与 `strategies/README.md` 均属于兼容或公开说明入口；稳定规则、目录边界、验证矩阵和策略体系边界必须先沉淀在本产品分脑

## 2. 硬规则
- 先接主脑，再接本分脑，再进入具体 body
- `daily_stock_analysis-main/brain/` 是本产品分脑 AI 入口；工作区级规则、分脑拓扑、默认接管顺序和全局分支纪律仍以主脑 `brain/` 为准
- `AGENTS.md / CLAUDE.md / SKILL.md` 属于兼容资产；若与主脑冲突，先服从主脑，再同步本分脑和兼容入口
- 新增配置项时要同步更新 `.env.example` 和相关说明
- 新增 README / docs 内容时，先判断是否为稳定事实、入口、流程或治理规则；若是，必须先写入本分脑对应中枢
- 面向 AI 接管的文档默认使用简体中文；公开多语言文档只作为产品用户入口，不替代 brain
- 本地项目提交按主脑 project profile 与 `tools.brain.project_commit` 执行；`git tag` 或 `git push` 仍需用户明确授权
- 不写死密钥、账号、端口、模型名、绝对环境路径或环境专属分支逻辑
- 用户可见行为、CLI/API、部署、通知或报告结构变化时，需要同步相关 docs / `docs/CHANGELOG.md`，并评估是否要写回 brain

## 3. 已验证教训
- 如果产品 AI 兼容资产不指向本分脑，后续 agent 会优先读错入口
- 如果模块地图和真实目录不一致，接管和维护会持续绕路
- 如果公开 README 的产品能力、配置约束和验证入口没有在 brain 中留痕，后续维护会把营销文案误当接管真源，或者漏掉关键配置边界
- 如果 AI 协作资产继续声明自己是唯一真源，而本分脑只保留摘要，后续 agent 会在 `brain-first` 和仓库原生入口之间反复摇摆；正确关系是主脑管理工作区，本分脑管理产品事实，仓库原生文档为兼容入口

---
applyTo: "main.py,server.py,src/**/*.py,data_provider/**/*.py,api/**/*.py,bot/**/*.py,tests/**/*.py"
---

# Backend Objects

`analysis_pipeline`: 主流程、服务层、仓储层、schema、API、bot 和测试共同构成后端对象。改动时优先复用当前服务、仓储、schema 和 fallback 逻辑。

`config_surface`: 配置、CLI flag、调度语义、API 行为、认证和报告载荷会影响本地运行、Docker、GitHub Actions、Web 与 Desktop。

`provider_surface`: `data_provider/` 保存数据源优先级、字段标准化、timeout、retry 和降级路径。

`validation_method`: 后端默认验证入口是 `./scripts/ci_gate.sh`；小范围改动可选择 `python -m py_compile <changed_python_files>` 加最近的确定性测试。

`optional_integration`: 单一 provider、通知渠道或可选集成失败通常进入降级路径；只有需求明确时才变成 fail-fast。

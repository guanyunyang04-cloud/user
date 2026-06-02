# Traditional Quant Research 状态中枢

- 当前状态：新建并已挂载到 workspace 主脑。
- 当前阶段：第一阶段脚手架完成，研究范围已定为上证主板 A 股与深证主板 A 股，并剔除创业板、科创板、ST、停牌、退市等标的；v2 PIT 数据集和默认研究框架已确定，下一步跑通基础传统因子到回测日志的第一条研究闭环。
- 已有骨架：`traditional_quant_research/` 核心函数、`tests/` 基础测试、`experiments/` 实验入口、`research_log/` 研究日志、`data/` 数据契约。
- 默认优先级：先构造沪深主板普通 A 股日频 OHLCV+复权因子数据集，再建立收益/因子/组合/回测最小基线。
- 默认研究框架入口：`brain/references/research_framework.md`。
- 默认分支纪律：repo-tracked mutation 优先在 `main` 分支执行。

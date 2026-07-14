# Quant Data Platform

Canonical brain source: `quant_data_platform/brain/identity_layer.md`.

主脑管辖下的共享量化数据平台分脑。本文只保留项目入口；当前 v2 active、v3 rebuild、安全阻断、数据契约和 provider 对象以 `quant_data_platform/brain/` 为准。

## Entry

- 项目身份：`quant_data_platform/brain/identity_layer.md`
- 当前状态：`quant_data_platform/brain/state_center.md`
- 过程入口：`quant_data_platform/brain/operations_center.md`
- 参考文档：`quant_data_platform/brain/references/`

## Body

- `src/`：平台代码。
- `configs/`：canonical 数据域与 profile 配置。
- `registry/`：可审计 manifest、registry 和指针。
- `data/`：大数据资产、tmp、memmap、agent runs，默认不进 Git。
- `tests/`：平台测试。

## Current Boundary

- `data/qdp_v2` 仍是唯一有效、可研究的 active。2026-07-15 发现的空 v3 active 已确认无候选/无数据并删除；默认 status 已恢复为 v2 `status=ok`、as-of `2026-06-26`。
- v3 当前合同是 `qdp_v3_20260715_trusted_source_5m` / schema `3.3.0` / manifest `4`；5m 是唯一分钟域，首次发布只要求九个市场核心域。
- v3 历史基底固定使用 Tushare proxy `2010-01-01..2026-07-13`；截止日后才由 BaoStock 更新日线/状态/因子事件，mootdx 更新 5m，BaoStock 仅作完整股票日 fallback。
- 历史流程不再运行 BaoStock/Tushare 全量交叉验证；发布仍保留 schema、主键、identity、PIT、48 根 5m 和原子 CAS 闸门。
- runtime job/cursor/heartbeat 与 OS advisory lock 位于 `QDP_RUNTIME_ROOT`；同一进程三个 Tushare worker 共享 `96 rpm / burst 1` limiter。5 分钟额度优先，耗尽后只补 `status/factor`。
- `QDP_DATA_ROOT` 位于 H，runtime root 位于 C。H 已完成 `chkdsk /f` 并通过 clean/healthy/0-bad-sector 闸门；用户取消 F 盘备份。compact layout 为 reference=undated x 1、低频=year x 1、仅 5m=year x 16；legacy raw 只有在 bundle/index/hash round-trip 通过后才可删除。
- 首次 v3 发布不退休 v2；必须再成功完成一次 `2026-07-13` 后的增量发布，retirement gate 通过后才删除 v2 四条旧分钟链。
- 公共读取只在 v3 active 通过 manifest/候选/九域/published-at 验证后才切换到 v3，否则默认 v2；pytest 已隔离 production roots 和 credentials。任何 v2 写命令仍必须显式指定 `--generation v2`。
- 训练包和 memmap 归 `daily_research`，不再属于 QDP CLI。

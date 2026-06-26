# Quant Data Platform 身份层

## 身份
`quant_data_platform` 是主脑管辖下的共享量化数据平台分脑。

它不是独立 Git 仓，也不是团队平台项目；它是个人研究者工作区的数据大脑，负责把所有研究项目可复用的数据基底、registry、coverage audit、canonical bundle 和 memmap 治理收敛到一个清晰项目里。

## 北极星
- 建立唯一、可审计、可按日使用的 `canonical_data_v1` 数据基底。
- 让 `daily_research`、`traditional_quant_research`、`t0_project` 等项目只消费数据平台，不各自维护重复数据湖。
- 原始数据、派生特征、训练 profile 分层治理：canonical 负责完整承载，研究 profile 决定使用哪些字段。
- 效率优先：已完成的 canonical / sharded memmap 是默认基底；旧数据集、旧 bundle、旧单体 memmap 和旧兼容入口在确认非唯一后应清理或降级。

## 边界
- `quant_data_platform/data/` 承载大数据资产和运行产物，默认不进 Git。
- `quant_data_platform/registry/` 承载可审计 registry、manifest 指针和治理状态，可以进入 Git。
- `canonical_data/` 是过渡入口，不再作为长期主项目扩张。
- `daily_research` 不再拥有 data_platform / data_lake 公共能力；研究项目只能通过 QDP 公开 API/CLI 和已生成数据资产消费共享数据基底。

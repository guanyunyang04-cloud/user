# Quant Data Platform 过程目录

快照日期：`2026-07-16`

## Runtime Objects

`python`: `C:/Users/ASUS/miniconda3/envs/yolos/python.exe`

`workspace`: `H:\quant_project`

`data`: `H:\quant_project\quant_data_platform\data\qdp_v2`

`runtime`: `H:\quant_project\quant_data_platform\data\qdp_runtime`；任务完成并确认写入当前表后可直接删除。

`memory`: 不固定限制 DuckDB 为 2 GiB；可用内存低于 0.5 GiB 持续 5 秒才中断。

## Current Command Palette

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli status --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli list --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli describe market_intraday_5m --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli check --quick --json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.cli gc --dry-run --json
```

## Procedures

### procedure `update_recent_market`

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m quant_data_platform.qdp_v2.recent_market_repair all --start-date <date> --end-date <date> --workers 4 --workspace-root H:\quant_project
```

`steps`: mootdx 先快速补当前窗口；重新计算缺口；4 个 BaoStock 进程各维持一个独立登录，只补剩余股票日；仅完整 48 根日写入当前 5m 表；提交后删除 runtime。

### procedure `direct_data_change`

`steps`: 定位唯一当前 table；准备需新增/替换/删除的 Parquet；检查主键和最小结构；调用 QDP 原地 mutation API；确认 `dataset.json` 行数与 shard 路径；删除被替代 shard。

`boundary`: 不建立 candidate、副本或第二个 generation；修改 raw 值时保留一条简短 reason 即可。

### procedure `cleanup`

`steps`: 从 `active.json` 取得当前 14 个目录；删除其他 dataset generation、runtime、audit 和 provider staging；研究 event packs 不属于 canonical，不随 QDP GC 删除。

### procedure `verify`

`default`: `status` + `check --quick` + 相关域的主键/完整日小检查。

`full`: 只有怀疑大范围损坏时才运行全表扫描；普通更新不做跨源逐值比较。

## Validation Selection

- CLI/代码变化：focused pytest + `qdp --help` + `git diff --check`
- 当前表变化：active manifest/shard existence + 相关主键/行数/48-bar 检查
- 脑区变化：brain sync audit + doc guard + integrity check

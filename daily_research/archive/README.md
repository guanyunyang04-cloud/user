# Archive Rules

`daily_research/output/` 与 `daily_research/cache/` 都属于高频研究产物区，但两者角色不同：

- `output/`
  - 面向实验结果、人读汇总、对照表和日志。
  - 归档粒度按“实验目录”或“输出根目录文件”处理。
- `cache/`
  - 面向可重建或可复用的中间资产。
  - 归档粒度按“缓存桶内的哈希文件”处理，避免误搬整个共享缓存根目录。

## 热区与冷区

- 热区：
  - `daily_research/output/`
  - `daily_research/cache/`
- 冷区：
  - `daily_research/archive/output/`
  - `daily_research/archive/cache/`

冷区 payload 默认继续被 `.gitignore` 忽略，不进入 Git。
会被 Git 跟踪的只有：

- 本说明
- `daily_research/archive_policy.json`
- `daily_research/archive/manifests/` 下的归档清单

## 默认归档纪律

1. 先保留热区最近一段时间的结果，避免打断正在复盘的研究。
2. 只归档“超出保留天数且超出最近保留数量”的项目。
3. 若某条规则配置了 `max_hot_size_mb`，则：
   - `keep_recent_count` 与 `protect_globs` 仍然是硬保护；
   - `keep_recent_days` 会退化为软偏好，必要时会为了把热区压回预算而归档较旧的“近期文件”。
4. `cache/` 根目录下的 `industry_map_tq.csv`、`style_map_tq.csv` 这类基础映射文件不参与默认归档。
5. `deep_alpha/states`、`rolling_pools`、`liquidity_buckets` 这类体积较小、复用频繁的控制缓存默认不归档，后续如体积异常再单独加规则。

## 推荐工作流

先做 dry-run：

```bash
python daily_research/tools/workspace_maintenance.py archive
```

如只看某一类规则：

```bash
python daily_research/tools/workspace_maintenance.py archive --rules output_runs,deep_alpha_corpus_cache
```

确认无误后再执行移动：

```bash
python daily_research/tools/workspace_maintenance.py archive --apply
```

执行后会生成 manifest，默认写入：

- `daily_research/archive/manifests/archive_<batch>.json`

建议归档后顺手做两件事：

1. 用 `workspace_maintenance.py report` 复核热区体积是否下降。
2. 将新的归档 manifest 连同相关说明一起提交到 Git。

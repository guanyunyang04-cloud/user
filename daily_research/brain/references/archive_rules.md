# 归档规则

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

## 入口级归档

- 第二轮瘦身后，旧 `replay-based` 产物默认不再在 brain 主文档中散落直引。
- 当前统一入口：
  - `daily_research/archive/output/replay_based_reference_index.md`
  - `daily_research/archive/output/replay_based_reference_index.json`
- 这属于“入口级归档”，不等于立即物理搬运 payload；原始实验根仍保留在 `daily_research/output/`，但语义上统一降级为 `mechanism_reference_only`。
- 若旧 `replay-based` 读数与 corrected recent、current default 或最新 strongest verdict 冲突，一律以后者为准。

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

## 2026-05-16 接管减负批次

- 批次名：`handoff_simplification_20260516_01`。
- Manifest：`daily_research/archive/manifests/archive_handoff_simplification_20260516_01.json`。
- 执行命令：

```bash
C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/tools/workspace_maintenance.py archive --batch-name handoff_simplification_20260516_01 --apply
```

- 移动结果：`109` 项，约 `21.07GB`，主要来自 `advanced_ml` 与 `deep_alpha` 缓存，以及少量过期 output run。
- 本批次只做归档，不做删除；未运行 `clean --apply` 或 `prune-archive --apply`。
- 本批次不修改 `daily_research/output/active_execution_strategy.json`，不触发训练、评估、study、production refresh 或 live/default 切换。

### 本批次保留热区

- `daily_research/output/continuous_policy/`：当前 research / shadow evidence 热区，仍需 explicit tag 或 dataset id 读取。
- `daily_research/output/research_data_lake/`：当前 DuckDB + Parquet data lake 真源热区。
- `daily_research/output/short_expert_policy_v5b_execalign_production_default/`：当前 production root。
- `daily_research/output/active_execution_strategy.json`：当前 live/default 物化真源，禁止归档或修改。

### 后续复核命令

```bash
git diff -- daily_research/output/active_execution_strategy.json
C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/tools/workspace_maintenance.py report
C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/tools/doc_guard.py check
C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/tools/brain_integrity_check.py --json
git diff --check
```

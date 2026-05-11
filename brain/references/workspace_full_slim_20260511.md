# 全仓瘦身维护胶囊

日期：`2026-05-11`

## 当前裁决
- 本轮只清理本地研究产物、Git 跟踪卫生、低风险测试结构和 brain/tooling 守卫；未提交、未推送、未重写 Git 历史。
- `daily_research/output/active_execution_strategy.json` 未修改；production root、live/default/promotion 均未切换。
- 正式研究输出只归档到 `daily_research/archive/output/`，没有直接删除。
- 可重建 cache 先归档、再按 prune manifest 删除；没有 manifest 的 cache 未删除。

## 结果摘要
- 初始事实：`daily_research` 约 `182.84GB`，其中 `cache=94.07GB`、`output=88.75GB`。
- 第一批 archive：`20260511_full_slim_r1` 移出热区 `952` 项、`110.60GB`。
- 第一批 prune：按 `prune_20260511_full_slim_r1.json` 删除已归档 cache `449` 项、`34.22GB`。
- 第二批 policy 收紧：`deep_alpha/features` 热保留从 `12` 降到 `2`，`deep_alpha/corpus` 热保留从 `6` 降到 `1`。
- 第二批 archive：`20260511_full_slim_r2_cache_policy` 移出热区 `15` 项、`41.78GB`。
- 第二批 prune：按 `prune_20260511_full_slim_r2_cache_policy.json` 删除已归档 cache `15` 项、`41.78GB`。
- `pycache` 清理：dry-run 命中 `600` 项、`23.70MB`，随后执行删除。
- 清理后事实：`daily_research=106.84GB`、`archive=76.38GB`、`cache=18.06GB`、`output=12.38GB`。
- 实际释放磁盘约 `76.02GB`；热区移出规模约 `152.38GB`。

## Manifest
- Archive dry-run：`daily_research/archive/manifests/archive_dryrun_20260511_full_slim.json`
- Archive apply：`daily_research/archive/manifests/archive_20260511_full_slim_r1.json`
- Prune dry-run：`daily_research/archive/manifests/prune_dryrun_20260511_full_slim_r1.json`
- Prune apply：`daily_research/archive/manifests/prune_20260511_full_slim_r1.json`
- Cache policy archive dry-run：`daily_research/archive/manifests/archive_dryrun_20260511_full_slim_r2_cache_policy.json`
- Cache policy archive apply：`daily_research/archive/manifests/archive_20260511_full_slim_r2_cache_policy.json`
- Cache policy prune dry-run：`daily_research/archive/manifests/prune_dryrun_20260511_full_slim_r2_cache_policy.json`
- Cache policy prune apply：`daily_research/archive/manifests/prune_20260511_full_slim_r2_cache_policy.json`

## Git 卫生
- `a_stock_daily_selection/output/artifacts/*.pkl` 四个大型研究产物已从 Git index 移除，本地文件保留并被 `.gitignore` 阻止再次进入版本库。
- 保留轻量 manifest：`a_stock_daily_selection/output/artifacts/artifact_manifest_20260511.json`。
- 新增 tracked-large-file 守卫：`doc_guard.py check` 默认禁止未 allowlist 的大型 tracked `.pkl/.pt/.gif/.psd` 等文件。
- 当前 allowlist：`brain/tracked_large_file_allowlist.json`，仅保留 DSA 已有品牌资产。
- 深链规则后续不再单独跑全仓搜索；Codex 线程 URI 禁止由 `doc_guard.py check` 的 thread-deeplink 守卫覆盖。

## 代码结构
- 第一批行为保持拆分只移动测试夹具：`daily_research/continuous_policy/tests/portfolio_daily_fixtures.py`。
- `test_portfolio_daily_strategy_contracts.py` 保留原测试合同和调用名，生产策略逻辑未修改。
- focused verification：`104 passed`。

## 保护路径
- `daily_research/output/active_execution_strategy.json`
- `daily_research/output/short_expert_policy_v5b_execalign_production_default`
- `daily_research/output/continuous_policy`
- `brain/` 与各分脑 `brain/`
- 最新 r52d explicit evidence / screening artifacts

## 遗留风险
- `daily_research/cache=18.06GB` 仍略高于维护阈值 `15.25GB`。
- `deep_alpha/corpus` 仍有最新单文件 `11.01GB`，超过 `4GB` 热预算；本轮保留，不硬删最新热集。
- `deep_alpha/features=3.28GB` 略高于 `3GB` 热预算；保留最近两个热缓存。
- `deep_alpha/raw=2.12GB` 略高于 `2GB` 热预算。
- `daily_research/output=12.38GB` 主要来自当前 `continuous_policy` 证据与最新生产/研究入口；本轮不再归档。
- `daily_research/archive/output=76.38GB` 仍占本机磁盘；若需要继续释放空间，应迁移到外部冷存储或另行审批删除正式输出。

## 回滚方式
- 已归档 output 可按 archive manifest 的 `source_path` / `archive_path` 对照移回原位。
- 已 prune 的 cache 不应从 Git 或 brain 恢复；需要时重新运行对应研究流程重建。
- Git index 中移除的 `.pkl` 若必须短期恢复跟踪，可用 Git 恢复索引，但不建议重新提交大型 pickle。
- 若新 archive policy 过紧，可把 `deep_alpha/features.keep_recent_count` 与 `deep_alpha/corpus.keep_recent_count` 调回旧值；这不会恢复已 prune cache，只改变后续保留策略。

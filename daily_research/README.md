# daily_research

Canonical brain source: `daily_research/brain/identity_layer.md`.


`daily_research` 是当前工作区的正式生产研究与执行主线，负责长期研究、连续策略、默认执行、Web 控制台、维护工具和项目分脑状态。

当前工作区根目录：`H:\quant_project`。`daily_research` 不再运行在 `H:\new_tdx64\PYPlugins\user` 下，旧通达信插件用户目录不应保存本项目代码或产物。

接管默认先从用户目标、当前 git 状态和相关主脑 / 分脑事实判断；需要机器可读上下文或守卫提示时，可从工作区主脑 capsule 获取诊断摘要：

```powershell
C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow capsule --task "<task>" --json
```

确认任务属于 `daily_research` 后，`daily_research/brain/` 是本项目事实层。本 README 只作为简体中文快速索引，不替代主脑或分脑中的当前对象、过程、证据和治理判断。

## 模块地图

- `baseline/`：历史研究链路与仍受支持的交易计划管线
- `continuous_policy/`：当前连续决策策略的训练、评估、导出与协议编排
- `path_policy/`：多 Horizon 交易效用排序、历史 Path20 预测/RL 与 shadow 对照研究
- `deep_alpha/`：更长周期的模型架构、alpha 与执行策略研究
- `execution/`：执行应用、任务运行器、Web 控制台与 production 更新入口
- `tools/`：守卫、报告、维护工具与一致性检查
- `brain/`：当前对象、长期知识、治理不变量与过程记忆
- `output/`、`cache/`、`archive/`：生成产物、热缓存与冷归档

## 环境

标准环境是 `yolos`，依赖真源为 [environment.yml](daily_research/environment.yml)。

本地数据边界：
- `daily_research` 正式研究链路不再依赖 `t0_project/tqcenter.py`、`pytdx` 或 `mootdx`。
- QDP 是共享数据基底唯一 owner；`daily_research` 只消费 QDP 输出的 lake、manifest、memmap 和 training pack。
- 在线 provider 每日更新、CSV 导入、canonical bundle、pool/sector-board view 和数据审计都通过 `quant_data_platform` CLI/API 完成。
- `csv` 只作为 QDP 入湖导入或补洞通道，不允许被正式训练/评估直接读取。
- QDP 数据刷新支持 `--universe all_a|liquid500|file:<path>|symbols:<csv>`；`--symbols` 只保留为小样本/显式调试入口。

## 常用入口

确认本任务属于 `daily_research` 后，再使用以下项目入口；capsule / route 可辅助判断，但不替代 agent 对用户目标和文件证据的判断：

- 当前 seq100 path-value 主线：
  `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline contract --json`
- 当前主线单模型训练入口（兼容/诊断用途，不能单独形成正式 verdict）：
  `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train --json`
- 当前主线 summary_v2 对照入口：
  `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-summary-v2 --json`
- 当前主线 summary_v2 no60 严格对照入口：
  `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-summary-v2-no60 --json`
- 当前主线 all-channel summary_v2 组合对照入口：
  `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-summary-v2-price-delta --json`
  `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-summary-v2-ohlcva-aux --json`
  `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-summary-v2-ohlcva-aux-low --json`
  `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-summary-v2-ohlcva-aux-low-price-delta --json`
  `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-summary-v2-ohlcva-path-equal --json`
- 当前主线 daily-only no-minute 输入消融入口：
  `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-daily-only --json`
- 当前主线 daily-only + summary_v2 组合对照入口：
  `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-daily-only-summary-v2 --json`
- 当前主线 price-delta 对照 / 默认低权重 OHLCVA 辅助入口：
  `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-daily-only-summary-v2-price-delta --json`
  `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-daily-only-summary-v2-ohlcva-aux-low --json`
  `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-daily-only-summary-v2-ohlcva-aux-low-price-delta --json`
- 当前主线 OHLCVA 六字段等权 path-loss 对照入口：
  `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-daily-only-summary-v2-ohlcva-path-equal --json`
- 当前主线 purged 多年 walk-forward 入口：
  `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_walkforward build-folds --oos-years 2022-2025 --json`
  `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_walkforward run-study --oos-years 2022-2025 --profiles summary_v2_all_channels,daily_only_summary_v2_ohlcva_aux_low --seed 7 --epochs 1 --device cuda --json`
- 正式 fold views：
  `daily_research/data/research_store/views/seq100_path60_todayclose_ohlcva_purged_oos{2022,2023,2024,2025}.json`
- 旧 2025 duplicated validation/test view 仅保留历史复现，不再用于正式判断：
  `daily_research/data/research_store/views/seq100_path60_todayclose_ohlcva_train2012_2024_val2025_test2025.json`
- 当前主线部分分钟线输入消融入口：
  `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-no-intraday-summary --json`
  `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-no-limit-structure --json`
- 当前主线 direct-value 排序对照入口：
  `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-direct-value-5d --json`
  `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-direct-value-10d --json`
  `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.path_policy.seq100_mainline train-direct-value-60d --json`
- 连续策略正式协议：
  `python daily_research/continuous_policy/run_continuous_policy_protocol.py ...`
- TDX-free V2 每日数据刷新：
  `python -m quant_data_platform.cli refresh-daily --as-of-date YYYY-MM-DD --provider-plan default_free --universe all_a --domains market_daily,trading_calendar,universe_snapshot,security_status,limit_status,industry_concept,valuation --json`
- CSV 入湖导入：
  `python -m quant_data_platform.cli import-csv --input <csv_or_folder> --domain market_daily --as-of-date YYYY-MM-DD --source-name manual_csv --json`
- 执行应用：
  `python daily_research/execution/run_execution_app.py run --task <task-name> -- ...`
- 执行 Web 控制台：
  `python daily_research/execution/run_execution_web.py`
- 工作区维护报告：
  `python daily_research/tools/workspace_maintenance.py report`

当前默认研究概念面已收窄为 `seq100_x32_daily_input -> today_close_anchor -> future60_ohlc_path -> summary_v2_multi_horizon_ohlc -> low_weight_va_auxiliary -> path_trade_value_v2 -> path_value_spread`。`daily_only_summary_v2_ohlcva_aux_low` 是当前默认研究入口：使用 `daily_raw + daily_state` 32 维输入、`5/10/20/40/60` 多窗口 OHLC 派生 summary loss，并以低权重 VA 辅助监督 (`va_level=0.02`, `va_delta=0.01`) 约束量价表征；summary/value/rank 仍保持 price-only OHLC 派生 `path_trade_value_v2`。正式评价已改为 2022-2025 purged expanding `train/oos`：每条训练标签必须在 OOS 首日前结束，归一化不使用 OOS feature dates，训练期间不读取 OOS，固定 final epoch 后只评估一次。旧 `2024 validation + 2025 train-through-2024 forward test` 含标签跨年问题，降为 pre-purge 历史证据。新四年结果中，all-channel 和默认 profile 的 Top3 alpha 点估计都四年为正；all-channel fold-equal 均值略高 (`10.03%` vs `9.38%`)，但配对差仅 `+0.65 percentage points (0.0065)`，HAC 与连续有序 969 日 non-circular 60 日块 bootstrap 区间均跨 0；按年 grouped bootstrap 仅作敏感性检查。2023 又明显反向，因此不替换更简单、worst-fold 更强的 daily-only 默认。完整结论见 `daily_research/brain/references/path_policy_seq100_purged_walkforward_2022_2025_result_20260710.md`。旧 `all_channels_base_summary` 保留为 broad-TopK 对照基线；`summary_v2_no60`、输入消融、direct-value rank、`alpha_v2`、`path20`、`symbol_embedding`、`residual_score`、`richer_target`、`ohlcva_unified` 和 `rank_heavy_top1` 默认只作为历史、对照或暂停分支。

真实运行时优先使用显式 `yolos` Python，例如：

```powershell
C:\Users\ASUS\miniconda3\envs\yolos\python.exe -X utf8 daily_research\tools\project_consistency_check.py --mode research
```

## 验证

较大改动前后建议运行：

```powershell
python -m compileall -q daily_research
C:\Users\ASUS\miniconda3\envs\yolos\python.exe -m tools.brain.integrity_check --json
C:\Users\ASUS\miniconda3\envs\yolos\python.exe -X utf8 daily_research\tools\project_consistency_check.py --mode research
C:\Users\ASUS\miniconda3\envs\yolos\python.exe -m tools.brain.doc_guard check --scope changed
```

执行端解冻、active artifact 或 live-facing 合同变更时，再显式运行 `project_consistency_check.py --mode execution` 或 `--mode full` 与裸 `doc_guard check`。

## 治理

- 默认按用户目标、路径、文件证据和主脑边界判断是否进入 `daily_research/brain/`；capsule / route 只是可选诊断信号。
- 工作区根 `brain/brain_manifest.json` 定义所有分脑共享的主脑合同。
- 新的长期事实先整合进对应 brain；README 只保留简体中文索引和公开入口。
- 生成实验产物应留在 `daily_research/output/`，需要复核或裁剪时使用 `workspace_maintenance.py`。

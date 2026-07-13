# Seq100 Q-only 受控对照提前终止结果

## 结论

Seq100 Q-only 研究于 2026-07-13 按用户决定提前结束。新 study 的 12 个逻辑任务最终为 `10 completed / 2 cancelled`，不再运行正式 selection，`winner=null`，不 freeze、不 promotion，QDP active 与 active execution 均未改变。

提前结束不是用未完成折推断 2024/2025 表现，而是冻结冠军门槛已经不可达：`qcurve_multiscale_ma_qonly` 已完成的 2022、2023 两年 Top3 成本后绝对收益均为负；即使未跑的 2024、2025 全部为正，也最多达到 `2/4`，低于合同要求的 `3/4`。LGBM 与 GRU Q-only 也已确定不合格，因此继续消耗 GPU 不会产生合格 winner。

## 研究合同与资产

- 代码基线：`e9ac92874295aac7571f1909cd59675579d37ed8`。
- 数据：candidate-complete v8，只读；未重建 v7/v8。
- 新 study：`daily_research/output/path_policy/studies/seq100_dynamic_qcurve_qonly_2022_2025/`。
- 冻结 seed：7；2022-2025 purged expanding development；每折独立按 development loss 早停。
- Q-only profile：`qcurve_gru_qonly`、`qcurve_multiscale_ma_qonly`；LGBM 四折从旧 study 导入。
- Q-only loss 只保留 mean、quantile、positive、soft-exit、full-day rank 五项；没有 OHLCVA 辅助头或 future OHLCVA 读取。

## 已完成结果

表中收益均为状态化 max3 组合成本后结果；Top3 是逐日成本后 mean log return，q20 是 enter q20 实际跌破率。

| profile | year | best epoch | dev loss | base net | base alpha log | stress net | Top3 abs | Top3 alpha | q20 breach | capital use |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qcurve_lgbm | 2022 | 10 | 0.677479 | 0.00% | 17.75% | 0.00% | 0.0000 | 0.0000 | 20.40% | 0.00% |
| qcurve_lgbm | 2023 | 10 | 0.579716 | 0.00% | 5.18% | 0.00% | 0.0000 | 0.0000 | 17.27% | 0.00% |
| qcurve_lgbm | 2024 | 10 | 0.751376 | 0.00% | 6.65% | 0.00% | 0.0000 | 0.0000 | 21.43% | 0.00% |
| qcurve_lgbm | 2025 | 10 | 0.647378 | 0.00% | -20.18% | 0.00% | 0.0000 | 0.0000 | 8.42% | 0.00% |
| qcurve_gru_qonly | 2022 | 3 | 0.813587 | 41.31% | 52.33% | 33.40% | 0.0177 | 0.0124 | 22.29% | 21.24% |
| qcurve_gru_qonly | 2023 | 4 | 0.730452 | 0.00% | 5.18% | 0.00% | 0.0000 | 0.0000 | 15.02% | 0.00% |
| qcurve_gru_qonly | 2024 | 5 | 0.870614 | -57.65% | -79.26% | -62.46% | -0.1434 | -0.1551 | 25.82% | 69.52% |
| qcurve_gru_qonly | 2025 | 3 | 0.731318 | 16.22% | -5.14% | 14.62% | 0.0094 | 0.0040 | 10.26% | 5.68% |
| qcurve_multiscale_ma_qonly | 2022 | 3 | 0.781768 | 0.94% | 18.68% | -14.88% | -0.0043 | 0.0090 | 23.33% | 96.09% |
| qcurve_multiscale_ma_qonly | 2023 | 3 | 0.696375 | -8.48% | -3.68% | -16.55% | -0.0231 | 0.0101 | 25.63% | 92.49% |

### 解释

- LGBM 四年均不入场，绝对收益为零，不能通过正收益年份门槛。
- GRU Q-only 的 2022 很强，但 2023 全现金、2024 大幅亏损、2025 虽有绝对收益但 alpha 为负；q20 跌破率从 10.26% 到 25.82%，跨年校准不稳定。
- Multiscale Q-only 在 2022/2023 都高资金利用、高换手，却连续出现负 stress 与负 Top3 绝对收益。它学到了可下降的 development loss，但这个 loss 没有形成稳定的成本后选股优势。
- 删除辅助任务没有解决核心问题；当前证据更支持“Q/policy 目标与可执行排序仍不够稳健”，而不是“辅助头分走容量”是唯一瓶颈。

## 取消任务

- Multiscale/2024 在 epoch 1 完成 150/2749 个训练日后按用户指令终止；存在 epoch 内 checkpoint，但 `result_valid=false`，禁止自动恢复。
- Multiscale/2025 未启动，直接标记 cancelled。
- 不生成 multiscale deployment epoch；GRU 四折 best epoch 为 `3/4/5/3`，但本研究不部署，因此不继续发布固定 deployment 模型。

## 吞吐与可靠性结果

- 旧 deterministic TCN 代表日约 46.91 股票/秒，外推 2022 单 epoch 约 29.8 小时。
- fast cuDNN + AMP + 微批 1024 的预检达到约 2859 股票/秒，外推约 29.3 分钟，门槛通过；正式稳态通常约 5700-6200 股票/秒。
- epoch 内 checkpoint 恢复 smoke 通过；2023 在一次内存保护终止后从 epoch 4 offset 550 恢复，没有重复训练日。
- 日窗口构造的两份 host 临时数组已改成单份 gather；真实缓存三档逐元素完全一致，理论峰值减半。修复后的 2023 恢复日志最低可用内存 1.64GiB。

## 验证边界

- Q 曲线聚焦测试：`18 passed`。
- Python 编译与 `git diff --check`：通过。
- 因用户提前终止，本轮没有执行原计划的完整 `daily_research/path_policy/tests` 回归、正式 selection、两个 profile 的完整 deployment epoch，也没有声称 12/12 完成。
- QDP active SHA-256 仍为 `E56F72A6CBA8BCF86055817F6A0EC5E7391271FB3C27B4D628C3ABC62944051E`；active execution 文件前后均不存在。

## 后续边界

该 Q-only/动态 Q 曲线方向进入已终止研究证据，不再自动续跑。若未来重新研究，必须先提出能改变可执行 Top3 排序与跨年校准的新假设，建立新的 additive study；不能恢复本 study 的 2024 checkpoint 并把混合结果当作同一完整矩阵。

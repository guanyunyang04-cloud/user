# Current workspace state

Updated: `2026-07-27`

- QDP is usable with `active_as_of_date=2026-07-21`. PIT main-board daily history
  includes historical ST, long suspensions, and delisted securities.
- Daily domains are normalized to `symbol_history` effective ticker intervals
  for both code-change identities. The 5-minute dataset was not rewritten.
- Protected QDP datasets, research packs, and registered checkpoints remain in
  place. The stable model registry still contains 15 bundles.
- Daily Research's frozen registered baseline is Structured 180x35 V2 (`L35V2`),
  but the complete-PIT frozen audit invalidated its legacy selection evidence:
  survivorship-complete 2023-2025 ending equity was 73.52% below the legacy-pool
  replay, and the six-year PIT durability account was nearly wiped out by an
  omitted delisting.
- The deterministic signal-close 2x2 study is complete. `V2C-P0` won with
  `Top1 / 1 slot / fixed D44`: 2023-2025 liquidated equity was CNY 9.10m,
  annualized log growth was 0.76563, and all three annual log-growth values were
  positive. The formal model registry and active execution remain unchanged.
- All 12 successful checkpoints, full-candidate predictions, 4,148 account jobs,
  reports, and study runners remain under the ignored study output for later
  reproduction. Probability, OOF-tree, and 2026 work require separate contracts
  and have not started.
- The complete PIT 180x35 pack and six 2020-2025 fold views are built with a
  60-trading-day purge. No long training task is currently running.
- `seq100_pit_signal_quality_v1` is CLOSED as `research_design_insufficient` on
  2026-07-26. The active contract
  `daily_research/studies/seq100_pit_signal_quality_v1.json` was deleted by
  closeout; the archived byte-identical copy is
  `daily_research/research_records/seq100/seq100_pit_signal_quality_v1/contract.json`
  with compact record `artifact.json` beside it
  (`artifact_sha256=7a0ff473...4636e7`). The terminal record is
  `design_invalidation/attempt_001/design_invalidation.json`
  (`design_invalidation_sha256=26a92e69...b461`), defect class
  `objective_cannot_answer_stated_question`, six bound evidence files, ten
  recorded inspected alternatives, `burned_fold_years=[2023,2024,2025]`, four
  trained cells with three completed, `retained_as=architecture_exploration_evidence`,
  and `successor.must_be_new_study_contract=true`. All four frozen artifact
  hashes are preserved unchanged inside it. All closeout gates passed.
  Formal fold training was 3 of 18 cells done (LightGBM, all three folds, seed 7);
  one TabM cell was interrupted and resumable; there was no `evaluate` and no
  winner. Neural runtime v2 was implemented and hardware-qualified. The later
  independent learnability audit and short-horizon reassessment now supply a
  target decision, but no successor implementation contract or account policy
  is designed. The owner explicitly accepts reused 2023-2025 evidence for that
  decision without claiming a pristine holdout. Authoritative detail lives in
  `daily_research/brain/state.md` and
  `daily_research/brain/references/seq100_signal_quality_target_diagnosis_20260726.md`.
- The true-batch-1024 comparison remains paused and may resume only on explicit
  request. There is no active frontend or execution system.
- Repository simplification evidence remains
  `brain/references/repository_simplification_20260722.md`; the portable
  `workspace-brain/v1` manifests remain the takeover and protection map.
- The 2026-07-26 over-engineering retirement is complete, verified, and committed
  together with the follow-up gap work: 48 files changed, 1,880 insertions,
  11,121 deletions, plus 114 `tmp/` process artifacts untracked and gitignored
  with files left on disk. No protected dataset, research pack, or registered
  checkpoint was modified. Evidence is
  `brain/references/repository_simplification_20260726.md`.
- Both designed gaps are now implemented, verified, and committed. QDP has
  `qdp repair patch|mutate|replace-table` inside the existing `repair.py`,
  dry-run by default and requiring `--apply`, `--reason`, and
  `--expected-manifest-sha256`, with append-only in-manifest receipts carrying
  reason, previous manifest hash, affected date range, changed columns, both
  schema hashes, and a UTC timestamp. `dataset.json` `schema`/`schema_hash` is now
  authoritative and `qdp check --quick` verifies footer schema and column order.
  Three of the 14 active manifests carry legacy declarations
  (`security_identity` and `symbol_history` have `dtype` but no `type`;
  `trading_calendar` uses lowercase `string`/`bool`), so comparison is normalized
  through `canonical_manifest_schema()` and untyped declarations degrade to a
  name-and-order check plus a `legacy_untyped_manifest_schema` warning. Current
  result: status ok, 0 errors, 2 warnings, 14 datasets.
- Backfilling those legacy declarations was blocked by the closed study binding
  them, and is now unblocked. `seq100_pit_signal_quality_v1` bound
  `manifest_sha256` and `schema_hash` for all 14 domains and
  `seq100_signal_quality.py:554/559` raised on any change, with
  `protection.failure: stop_immediately_on_any_hash_change`. All 14 schema hashes
  currently match. Normalization becomes legal only after the rebind channel
  exists.
- Step 3 remains unstarted and needs owner sign-off: a study-scoped `rebind-qdp`
  in `seq100_signal_quality.py` that walks the receipt chain, appends a
  reason-bearing transition under `qdp_binding_amendments`, has
  `validate-research` verify every transition, and refuses when a correction
  intersects the dates a pack consumed. No `--force` escape.
- That audit left two structural gaps, now designed and scheduled. First, QDP has
  no supported schema-evolution or one-cell-correction path: `repair.py` is
  library-only with no CLI, which is what caused dated one-shot migration modules
  to accumulate. The fix is `qdp repair patch/mutate/replace-table` inside the
  existing `repair.py`, in-manifest repair receipts, authoritative
  `dataset.json` `schema`/`schema_hash` with footer drift detection in
  `qdp check --quick`, and a study-scoped `rebind-qdp` that refuses when a
  correction intersects the dates a pack consumed. Second,
  `seq100_fold_contract.py` puts `mtime_ns` and absolute paths in its backing-file
  provenance digest, so copying or restoring a pack invalidates every fold
  contract; identity becomes pack-relative path plus size plus content SHA-256 at
  `source_view_provenance.schema_version = 2`, with a narrow v1 reader so the six
  recorded `l35v2_sixfold_2020_2025` contracts and protected training summaries
  keep their registered hashes.

<!-- seq100-signal-quality:start -->
- `seq100_pit_signal_quality_v1` 已以 `research_design_insufficient` 收口；selector winner 为 `null`。
- 权威 compact record：`daily_research/research_records/seq100/seq100_pit_signal_quality_v1/artifact.json`。
- `seq100_short_horizon_target_reaudit_v1` 已完成 15/15 个新 LightGBM 并收口。D3 pre-2023 固定 K3 最低跨窗口 ARI 为 0.7025，且与逐日 `g_3` 三分位 NMI 仅 0.1723，证明早期路径状态不只是终点收益离散化。
- 第一版入场机会主标签确定为 D5/D10/D20/D40 `mfe_H`；早期路径保留 D3/D5/D10 `state_H`，D20 state 降为次级对照；风险输出保留 D3/D5/D10/D20/D40 `pre_peak_mae_H`。`g_1/g_3` 拒绝，`g_60` 只保留为长期挑战者。
- 这支持“两阶段”语义：入场前预测异常大的上涨机会幅度，入场后依据实际早期路径更新留存/退出判断。各期限标量与状态概率暂时分开输出，不先合并为一个向量损失。
- 权威 compact record：`daily_research/research_records/seq100/seq100_short_horizon_target_reaudit_v1/artifact.json`；解释：`daily_research/brain/references/seq100_short_horizon_target_reaudit_20260727.md`。用户授权复用 2023-2025 作为本轮确认/决策窗口，但不宣称它们是全新留出集；2026 完全未读取，也不是本轮标签决策的必要确认年。
- 正式模型 registry、active execution、QDP 与完整 PIT pack 均未修改；下一步只能另立合同做保留标签的分组特征增量审计，尚未选择退出规则、持有期、槽位、杠杆、止损或继任架构。
<!-- seq100-signal-quality:end -->

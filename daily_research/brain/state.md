# Daily Research state

Updated: `2026-07-24`

- The registered frozen baseline remains `structured_joint_turnover_180x35_v2`
  (`L35V2`), but it is no longer valid selection evidence for deployment. The
  complete-PIT frozen audit found severe survivorship bias and a legacy
  next-open denominator distortion.
- The formal audit is
  `daily_research/research_records/seq100/pit_l35v2_survivorship_frozen_audit_2020_2025/`.
  For the legacy-selected Top1/1 + D14 strategy, 2023-2025 liquidated ending
  equity fell from CNY 52.18m to CNY 13.82m on the complete PIT universe; the
  2020-2025 replay ended at CNY 21.48 after a zero-recovery delisting.
- The complete PIT 180x35 pack is
  `daily_research/data/research_store/seq100_pit_l35v2_v1/`. It contains
  8,531,565 samples over 3,419 securities and six development fold views. The
  training purge is 60 trading days; execution-tail observations are not model
  targets.
- The deterministic signal-close 2x2 study is complete. `V2C-P0` won with
  `Top1 / 1 slot / fixed D44`; CNY 1m became CNY 9.10m after liquidation,
  annualized log growth was 0.76563, maximum drawdown was -48.53%, and all three
  annual log-growth values were positive. Three of four arms qualified.
- All six P1 folds passed the per-step Rank-gradient budget evidence. All 12
  successful checkpoints, predictions, account jobs, reports, and runners are
  retained under the ignored study output for reproduction.
- Probability variants, strict OOF tree reranking, and 2026 confirmation have
  not started and require separate contracts. The true-batch-1024 study remains
  paused and resumes only on explicit request.
- The old V4 pilot is diagnostic evidence only. It fixed the fictitious low-open
  path but selected pre-signal overheated names whose next-open execution often
  preceded mean reversion; it must not be resumed under its old contract.
- 2026 remains frozen confirmation only. No 2026 result may select a model,
  ranking rule, exit rule, Top-K, slot count, or account behavior.
- Protected packs and the 15 registered model bundles remain unchanged. The
  retained study checkpoints are research evidence and are not registered or
  active for execution.

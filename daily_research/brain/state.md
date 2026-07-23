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
- The only active route is
  `daily_research/studies/signal_close_path_value_2x2_v1.json`: deterministic
  V2C-P0/P1 versus V4-P0/P1 on the same pack, architecture, seed, folds, and
  account contract. V2C keeps V2 penalties and legal exits while anchoring the
  proxy path to signal-day close.
- P1 training is gated on a continuous dynamic rank-gradient budget. The old
  first-batch calibration followed by a frozen rank weight is forbidden.
- Probability variants and strict OOF tree reranking are deferred until one
  deterministic semantic qualifies in at least two of three 2023-2025 folds.
  The true-batch-1024 study remains paused and resumes only on explicit request.
- The old V4 pilot is diagnostic evidence only. It fixed the fictitious low-open
  path but selected pre-signal overheated names whose next-open execution often
  preceded mean reversion; it must not be resumed under its old contract.
- 2026 remains frozen confirmation only. No 2026 result may select a model,
  ranking rule, exit rule, Top-K, slot count, or account behavior.
- Protected packs and registered checkpoints remain unchanged. No training was
  started during the 2026-07-24 takeover.

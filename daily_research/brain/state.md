# Daily Research state

Updated: `2026-07-22`

- Current baseline: `structured_joint_turnover_180x35_v2`.
- Registered vintages: Legal flat and Structured 100×32 for 2023–2026; L35V2
  for 2020–2026.
- Current compact evidence: `daily_research/research_records/seq100/index.json`.
- Completed durability evidence: L35V2 folds for 2020–2025. All six Rank ICs
  are positive (`0.0838–0.1136`). Top1/1 + D14 maximizes six-year growth but
  has a negative 2022; Top3/3 + D42 is the highest-growth strategy with all six
  years positive; Top3/3 model-plan is the strongest autonomous exit.
- Paused work: test the same L35V2 recipe with true batch 1024.
- Retired routes: global-tail, intraday inputs, Capital Speed V3, bounded training
  windows, Q-curve branches, historical alpha-v2/path20 frameworks, and all old
  frontend/execution code.
- The latest preserved read-only signal evidence uses signal date 2026-07-21.
- Old generated output, duplicate checkpoints, terminal experiment runners, and
  layout-compatibility tests were removed after stable model/evidence migration.
  Fold identity validation now lives in the small shared
  `seq100_fold_contract.py` module instead of an experiment orchestrator.

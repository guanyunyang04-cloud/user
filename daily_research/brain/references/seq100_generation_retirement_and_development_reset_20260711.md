# Seq100 Generation Retirement And Development Reset

Date: `2026-07-11`

Status: `user-approved protocol replacement / old generation retired / successor implementation in progress`.

## Retired generation

The study at
`daily_research/output/path_policy/studies/seq100_pit_adjusted_global_tail_generation_20260711_v1`
is retired and must not resume. Its immutable retirement record is
`study_retirement.json` with retirement SHA-256
`345a4c3617079d52ca93e82b0aea985a3af00442811c72d521767e05f8a2fa16`.

The retained evidence is exactly:

- screen `12/12` registered;
- confirmation `3/24` registered;
- one unregistered partial job, `confirmation:baseline:oos2019:seed7`;
- no frozen champion;
- no outer claim or outer result;
- QDP active SHA-256 unchanged at
  `E56F72A6CBA8BCF86055817F6A0EC5E7391271FB3C27B4D628C3ABC62944051E`;
- active execution unchanged.

The retired code-provenance SHA-256 is
`42fcf4e166d2d00169192c284347e02a063343cc68c88da1947bc99de27a27a5`.
Later code changes belong only to the successor generation and invalidate any
attempt to append results to the retired registries.

## Approved successor method

The successor contract id is
`seq100_candidate_complete_development_walkforward_contract_20260711_v1`.
Its method is:

- treat `2022..2025` as development/model-selection years, not outer/test years;
- use purged expanding chronological folds with canonical roles
  `train/development`;
- use seed `7` and all eligible fold training rows, without a sample cap;
- require at least one complete epoch;
- select checkpoints only by deterministic development total loss, with loss
  components retained as evidence;
- use Top1/3/5/10 and execution/portfolio metrics for model and loss-design
  decisions, not for within-run early stopping;
- retain every attempted candidate and rejection in one immutable ledger;
- after champion selection, retrain through `2025`, freeze all identities, and
  start the real forward lockbox from the freeze date in `2026`.

## Pre-training blockers

No successor model may train formally until all of the following pass:

1. The signal-day candidate universe is independent of future label validity.
2. Supervision availability is represented by explicit masks, not candidate
   deletion.
3. Entry, blocked exit, terminal outcome, and cost semantics are deterministic.
4. The new pack, candidate index, supervised index, and development folds have
   immutable identities and zero label-dependency overlap.
5. Validation-loss early stopping completes at least one full epoch, restores
   the best checkpoint, and records every component loss.
6. QDP active and active execution remain unchanged.

Old packs, folds, registries, ledgers, checkpoints, and results remain read-only
historical evidence and must not be overwritten or relabeled.

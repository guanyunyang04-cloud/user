# Quant Data Platform brain

QDP owns the canonical current data store. Its public model is deliberately
small: one current table per domain, `active.json` as the index, `dataset.json`
as the table manifest, and Parquet as storage.

The supported commands are `status`, `list`, `describe`, `check`, `update`,
`exclude`, `compact`, and `gc`. Legacy lake generations, public candidates,
memmap builders, import bridges, and research-pack ownership are not part of
QDP. Model-ready packs belong to Daily Research.

Dataset mutations require an explicit data task and a passing QDP status/check.
Code cleanup must never rewrite dataset files.

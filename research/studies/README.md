# Study specifications

Files in this directory are durable study specifications. They are not a
guarantee that the old workflow can still be rerun: many specifications refer
to generated outputs from an earlier research tree that was intentionally
retired.

Path values follow these rules:

- Current inputs are workspace-relative POSIX paths such as
  `research/studies/...`, `research/records/...`, or `data/qdp/...`.
- New output destinations belong under `runs/studies/...`; durable evidence in
  `research/records/...` is never an output target for a rerun.
- `legacy://...` identifies a deleted historical input. It preserves the
  original path for provenance and is not a readable filesystem path.

Do not replace a `legacy://` reference with a similarly named artifact unless
the artifact identity and contract have been independently verified.

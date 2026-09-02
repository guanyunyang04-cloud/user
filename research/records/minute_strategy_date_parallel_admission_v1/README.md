# Date-parallel memory admission

The minute-strategy runner remains serial by default because each date worker
duplicates month-level context and DuckDB/pandas buffers. This record defines
the bounded parent-side admission policy for a future opt-in scheduler.

The implementation is
`src/quantlab/research/minute_strategy_admission.py`. Before starting a batch,
the parent calls `admit_date_workers(...)` with the number of already-active
workers. The calculation reserves:

- the non-negotiable 0.5 GiB machine floor;
- an additional 0.5 GiB transient-allocation margin;
- 3.5 GiB per worker, rounded upward from the full-month process-tree peak;
- an explicit maximum of two workers.

It returns the admitted *new* worker count and remaining budget. A scheduler
must call it again after a worker finishes or before adding another worker; a
child's own watchdog remains the final emergency guard.

## Evidence

The earlier two-date smoke test measured 1.79x wall-time speedup for two
concurrent 100-symbol processes, but available memory fell from 6.195 GiB in
the serial run to 3.791 GiB. The later full-universe April run reached 3.399
GiB process-tree RSS while remaining at least 3.552 GiB system-available. The
larger 3.5 GiB estimate is therefore intentionally conservative.

At the current machine snapshot, the helper generally admits one new worker;
two workers require roughly 8 GiB available before launch under this policy.
The policy is an admission guard, not proof that arbitrary full-universe date
workers are safe. A full parallel scheduler still needs a representative
multi-date benchmark and checkpoint/ordering handling before default enablement.

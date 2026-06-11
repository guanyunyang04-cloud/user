# Memmap Governance

The canonical registry lives in `quant_data_platform/registry`.

Experiments should first try the canonical registry before building a new forecast memmap. A registry hit is valid only when the signature matches the current canonical bundle, profile, universe, lookback, horizon, role years, and sample policy.

The old capped validation memmap remains useful as a smoke-test artifact, but after the canonical bundle changes it is stale for default reuse until a new memmap is built from the current bundle.

Full all-A feature storage must be sharded rather than a single huge `date x symbol x feature` file. The intended layout is yearly or year-symbol-block feature shards, separate label shards, and lightweight sample indices for experiments.

Cleanup stays dry-run only until the replacement bundle, registry, and memmap validation all pass.

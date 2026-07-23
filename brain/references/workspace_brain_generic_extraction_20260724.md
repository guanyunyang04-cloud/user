# Workspace Brain Generic Extraction 2026-07-24

## Decision

The globally installed `workspace-brain` is now a reusable project-memory
protocol, initializer, validator, and episodic capture tool. Its versioned source
lives outside this repository at `H:/codex_skills/workspace-brain`.

`quant_project` retains only its project brain instance: current identity and
topology, state, protected-object registry, child brains, and historical
evidence. Always-on project constraints remain in `AGENTS.md`.

## Migration

- Migrated the root and child manifests to `workspace-brain/v1`.
- Migrated the protected-object registry to `workspace-brain/objects/v1`.
- Removed the repository-local skill and global-skill hash coupling.
- Kept project-specific data, research, experiment, runtime, and Git rules local.
- Left existing historical references unchanged.

## Boundary

This change does not modify QDP datasets, the research store, model registries,
registered checkpoints, training semantics, or evaluation semantics.

## Verification

- The generic source suite passed `10` tests, including initialization,
  idempotence, path safety, child-cycle detection, capture isolation, and
  deployment drift cleanup.
- Three isolated forward tests covered a new CLI project, an existing
  `AGENTS.md`, and a protected canonical-data object. Their findings about ID
  truncation, warning counts, UTC calendar dates, and runtime cache copying were
  fixed before completion.
- The installed skill passed format validation, contains 11 source-matched
  files, and contains no repository-specific terms or Python runtime cache.
- The retained repository regression suite passed `256` tests.
- The generic doctor reported zero errors and zero warnings for all three
  project brains; the project integrity check returned `status=ok`.

# Brain-Native Superpowers Contract 2026-05-18

## Verdict
- Status: `brain-native workflow migration contract / API fallback / no plugin dependency`.
- Purpose: preserve useful Superpowers-style workflow discipline when API sessions cannot load the plugin.
- This is a brain operating-system enhancement, not a research mainline switch, strategy change, or active execution change.
- `daily_research/output/active_execution_strategy.json` must remain unchanged.

## Facts
- The existing brain already has the required foundation: task capsules, workflow registry, evidence registry, doc guards, brain integrity checks, and the `daily-research-brain` procedural skill.
- The brain now treats workflow discipline as a project-native capability:
  - `brainstorming_design`
  - `writing_plan`
  - `executing_plan`
  - `systematic_debugging`
  - `verification_before_completion`
  - `brain_writeback_verified`
- API and no-plugin sessions can start with:
  `C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.tools.brain_workflow capsule --child daily_research --task "<task>" --workflow auto --json`
- `workflow-guide` returns checklist, stop conditions, completion criteria, validation commands, and writeback routes for a workflow.

## Inferences
- This should reproduce most of the practical Superpowers benefit: planning discipline, debugging discipline, execution checklists, verification before completion, and guarded writeback.
- It cannot fully reproduce plugin-level behavior such as system-injected skill bodies or automatic trigger enforcement.
- The brain-native version is better aligned with this project because it includes project-specific active artifact guards, evidence registry rules, mainline switching rules, and daily_research command paths.

## Boundaries
- Do not copy the Superpowers plugin body into brain docs.
- Do not depend on the plugin being installed.
- Do not treat workflow output as brain truth; truth remains in current brain docs, references, and verified artifacts.
- Do not modify active execution without explicit future promotion authority.

## Next Allowed Actions
- Use `capsule --workflow auto` as the API bootstrap entrypoint.
- Add new project-native workflows only when they have explicit checklist, stop conditions, completion criteria, and guard commands.
- Keep workflow docs compact; put long evidence and dated contracts in `daily_research/brain/references/`.

# Generic Workflow Adapter Retirement 2026-05-25

## Verdict
- Status: `implemented / breaking cleanup / no compatibility shell`.
- The workspace brain no longer carries Superpowers-style generic workflow adapters.
- Local skills remain the authority for planning, debugging, TDD, verification, frontend, security, deployment, and general execution method.
- The brain remains the authority for project facts, routing, guards, evidence lookup, writeback routes, active artifact boundaries, and long-task monitoring tools.

## Removed
- Removed generic workspace workflow ids: `brainstorming_design`, `writing_plan`, `executing_plan`, `systematic_debugging`, `verification_before_completion`, and `long_task`.
- Removed stale daily_research references that described the old fallback model:
  - `daily_research/brain/references/brain_native_superpowers_contract_20260518.md`
  - `daily_research/brain/references/api_agent_bootstrap_prompt_20260518.md`

## Replacement Contract
- Start with `tools.brain.workflow capsule` for project routing and guard context.
- Use local skills for method discipline.
- Use `tools.brain.long_task_monitor` directly for PID/log/progress/artifact tracking of long-running work.
- For deterministic cleanup with clear benefit, low fact loss, and tests, remove stale paths fully instead of leaving compatibility shells.

## Validation
- `daily_research/output/active_execution_strategy.json` must remain unchanged.
- `doc_guard`, `integrity_check`, relevant workflow tests, and skill sync checks are the completion gates.

# API Agent Bootstrap Prompt 2026-05-18

Use this prompt when an API-only agent works on `daily_research` without the Superpowers plugin.

```text
Before substantial daily_research work:
1. Run:
   C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m daily_research.tools.brain_workflow capsule --child daily_research --task "<task>" --workflow auto --json
2. Read selected_workflow, workflow_guide, required_checklist, and stop_conditions.
3. Separate facts, inferences, assumptions, and action boundaries before acting.
4. Follow the workflow checklist.
5. Before any completion, fixed, passing, or safe claim, run fresh verification commands and read the output.
6. If evidence, mainline state, or contracts changed, write compact brain references and rebuild evidence_registry.
7. Never modify daily_research/output/active_execution_strategy.json without explicit promotion authority.
```

This bootstrap approximates plugin workflow discipline. It does not replace project truth, evidence artifacts, or active execution governance.

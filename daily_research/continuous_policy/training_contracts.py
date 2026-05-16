from __future__ import annotations

from typing import Any


TRAINER_BACKEND_PROTOTYPE_V1 = "prototype_gbdt_v1"
TRAINER_BACKEND_FORMAL_V2 = "formal_torch_v2"
TRAINER_BACKEND_FORMAL_SEQ_V3 = "formal_torch_seq_v3"
TRAINER_BACKEND_FORMAL_HIER_V4 = "formal_torch_hier_v4"
TRAINER_BACKEND_FORMAL_CORE_V4 = "formal_torch_core_v4"
TRAINER_BACKEND_FORMAL_PORTFOLIO_SET_V5 = "formal_torch_portfolio_set_v5"
TRAINER_BACKEND_FORMAL_DECISION_CORE_V6 = "formal_torch_decision_core_v6"
TRAINER_BACKENDS = (
    TRAINER_BACKEND_PROTOTYPE_V1,
    TRAINER_BACKEND_FORMAL_V2,
    TRAINER_BACKEND_FORMAL_SEQ_V3,
    TRAINER_BACKEND_FORMAL_HIER_V4,
    TRAINER_BACKEND_FORMAL_CORE_V4,
    TRAINER_BACKEND_FORMAL_PORTFOLIO_SET_V5,
    TRAINER_BACKEND_FORMAL_DECISION_CORE_V6,
)


def normalize_trainer_backend(value: str | None) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return TRAINER_BACKEND_PROTOTYPE_V1
    aliases = {
        "prototype": TRAINER_BACKEND_PROTOTYPE_V1,
        "gbdt": TRAINER_BACKEND_PROTOTYPE_V1,
        "gbdt_v1": TRAINER_BACKEND_PROTOTYPE_V1,
        "v1": TRAINER_BACKEND_PROTOTYPE_V1,
        "torch": TRAINER_BACKEND_FORMAL_V2,
        "torch_v2": TRAINER_BACKEND_FORMAL_V2,
        "v2": TRAINER_BACKEND_FORMAL_V2,
        "formal": TRAINER_BACKEND_FORMAL_V2,
        "seq": TRAINER_BACKEND_FORMAL_SEQ_V3,
        "seq_v3": TRAINER_BACKEND_FORMAL_SEQ_V3,
        "torch_seq_v3": TRAINER_BACKEND_FORMAL_SEQ_V3,
        "formal_seq_v3": TRAINER_BACKEND_FORMAL_SEQ_V3,
        "v3": TRAINER_BACKEND_FORMAL_SEQ_V3,
        "hier": TRAINER_BACKEND_FORMAL_HIER_V4,
        "hier_v4": TRAINER_BACKEND_FORMAL_HIER_V4,
        "torch_hier_v4": TRAINER_BACKEND_FORMAL_HIER_V4,
        "formal_hier_v4": TRAINER_BACKEND_FORMAL_HIER_V4,
        "core": TRAINER_BACKEND_FORMAL_CORE_V4,
        "core_v4": TRAINER_BACKEND_FORMAL_CORE_V4,
        "torch_core_v4": TRAINER_BACKEND_FORMAL_CORE_V4,
        "formal_core_v4": TRAINER_BACKEND_FORMAL_CORE_V4,
        "formal_torch_core_v4": TRAINER_BACKEND_FORMAL_CORE_V4,
        "portfolio_set": TRAINER_BACKEND_FORMAL_PORTFOLIO_SET_V5,
        "portfolio_set_v5": TRAINER_BACKEND_FORMAL_PORTFOLIO_SET_V5,
        "torch_portfolio_set_v5": TRAINER_BACKEND_FORMAL_PORTFOLIO_SET_V5,
        "formal_portfolio_set_v5": TRAINER_BACKEND_FORMAL_PORTFOLIO_SET_V5,
        "formal_torch_portfolio_set_v5": TRAINER_BACKEND_FORMAL_PORTFOLIO_SET_V5,
        "decision_core_v6": TRAINER_BACKEND_FORMAL_DECISION_CORE_V6,
        "torch_decision_core_v6": TRAINER_BACKEND_FORMAL_DECISION_CORE_V6,
        "formal_decision_core_v6": TRAINER_BACKEND_FORMAL_DECISION_CORE_V6,
        "formal_torch_decision_core_v6": TRAINER_BACKEND_FORMAL_DECISION_CORE_V6,
        "v4": TRAINER_BACKEND_FORMAL_HIER_V4,
        "v5": TRAINER_BACKEND_FORMAL_PORTFOLIO_SET_V5,
        "v6": TRAINER_BACKEND_FORMAL_DECISION_CORE_V6,
    }
    canonical = aliases.get(text, text)
    if canonical not in TRAINER_BACKENDS:
        raise ValueError(f"Unsupported continuous-policy trainer backend: {value!r}")
    return canonical


def build_training_contract(
    *,
    trainer_backend: str,
    runtime_env: str = "yolos",
    requested_epochs: int = 0,
    min_epochs: int = 0,
    resume_mode: str = "",
) -> dict[str, Any]:
    backend = normalize_trainer_backend(trainer_backend)
    resume_mode_text = str(resume_mode or "").strip().lower()
    if backend == TRAINER_BACKEND_FORMAL_CORE_V4:
        requested_epochs = max(int(requested_epochs or 0), 1)
        min_epochs = max(int(min_epochs or 0), 1)
        if not resume_mode_text:
            resume_mode_text = "strict"
        return {
            "trainer_backend": backend,
            "contract_class": "epoch_resume_shadow_research_candidate",
            "epoch_based": True,
            "promotable": False,
            "resume_capable": True,
            "gpu_required": True,
            "runtime_env": str(runtime_env or "yolos"),
            "min_start_epoch_budget": 1,
            "requested_epochs": requested_epochs,
            "min_epochs": min_epochs,
            "resume_mode": resume_mode_text,
            "notes": [
                "formal_torch_core_v4 is a parallel shadow-only research backend for release-first allocation.",
                "It is allowed to use epoch/resume GPU training, but it is not promotion-eligible by default.",
                "Promotion/live/default changes require a later explicit governance decision outside this backend contract.",
            ],
        }
    if backend == TRAINER_BACKEND_FORMAL_PORTFOLIO_SET_V5:
        requested_epochs = max(int(requested_epochs or 0), 1)
        min_epochs = max(int(min_epochs or 0), 1)
        if not resume_mode_text:
            resume_mode_text = "strict"
        return {
            "trainer_backend": backend,
            "contract_class": "epoch_resume_shadow_research_candidate",
            "epoch_based": True,
            "promotable": False,
            "resume_capable": True,
            "gpu_required": True,
            "runtime_env": str(runtime_env or "yolos"),
            "min_start_epoch_budget": 1,
            "requested_epochs": requested_epochs,
            "min_epochs": min_epochs,
            "resume_mode": resume_mode_text,
            "notes": [
                "formal_torch_portfolio_set_v5 is a parallel shadow-only research backend for portfolio-set release-first allocation.",
                "It uses temporal per-symbol encoding plus latent cross-sectional set attention; it is not promotion-eligible by default.",
                "Promotion/live/default changes require a later explicit governance decision outside this backend contract.",
            ],
        }
    if backend == TRAINER_BACKEND_FORMAL_DECISION_CORE_V6:
        requested_epochs = max(int(requested_epochs or 0), 32)
        min_epochs = max(int(min_epochs or 0), 32)
        if not resume_mode_text:
            resume_mode_text = "strict"
        return {
            "trainer_backend": backend,
            "contract_class": "epoch_resume_shadow_research_longrun_candidate",
            "epoch_based": True,
            "promotable": False,
            "resume_capable": True,
            "gpu_required": True,
            "runtime_env": str(runtime_env or "yolos"),
            "min_start_epoch_budget": 32,
            "requested_epochs": requested_epochs,
            "min_epochs": min_epochs,
            "resume_mode": resume_mode_text,
            "notes": [
                "formal_torch_decision_core_v6 is a shadow-only unified decision-core research backend.",
                "It must train against strict Gold targets and emit DecisionFrameV6-compatible predictions.",
                "Promotion/live/default changes are forbidden for this backend; evidence remains research shadow-only.",
            ],
        }
    if backend in {TRAINER_BACKEND_FORMAL_V2, TRAINER_BACKEND_FORMAL_SEQ_V3, TRAINER_BACKEND_FORMAL_HIER_V4}:
        requested_epochs = max(int(requested_epochs or 0), 32)
        min_epochs = max(int(min_epochs or 0), 32)
        if not resume_mode_text:
            resume_mode_text = "strict"
        notes = [
            "Formal continuous-policy candidates must run in yolos with CUDA enabled.",
            "Initial formal budget starts at 32 epochs; additional budget must continue via strict resume.",
            "Promotion evaluation is allowed only for this contract class.",
        ]
        if backend == TRAINER_BACKEND_FORMAL_SEQ_V3:
            notes.insert(
                1,
                "formal_torch_seq_v3 is the stronger temporal sequence branch used when v2 remains behaviorally constrained.",
            )
        if backend == TRAINER_BACKEND_FORMAL_HIER_V4:
            notes.insert(
                1,
                "formal_torch_hier_v4 is the hierarchical temporal portfolio branch with market/portfolio/cross-section interaction.",
            )
        return {
            "trainer_backend": backend,
            "contract_class": "epoch_resume_formal_candidate",
            "epoch_based": True,
            "promotable": True,
            "resume_capable": True,
            "gpu_required": True,
            "runtime_env": str(runtime_env or "yolos"),
            "min_start_epoch_budget": 32,
            "requested_epochs": requested_epochs,
            "min_epochs": min_epochs,
            "resume_mode": resume_mode_text,
            "notes": notes,
        }
    return {
        "trainer_backend": backend,
        "contract_class": "non_epoch_shadow_prototype",
        "epoch_based": False,
        "promotable": False,
        "resume_capable": False,
        "gpu_required": False,
        "runtime_env": str(runtime_env or "yolos"),
        "min_start_epoch_budget": 0,
        "requested_epochs": 0,
        "min_epochs": 0,
        "resume_mode": "",
        "notes": [
            "This backend is a non-epoch sklearn prototype used for lifecycle labeling and fast ablations.",
            "It may remain in shadow research and teacher generation, but it is not eligible for promotion.",
            "Epoch-based formal training rules do not apply until the run switches to a promotable backend.",
        ],
    }


def summarize_training_contract(payload: dict[str, Any] | None) -> dict[str, Any]:
    data = dict(payload or {})
    return {
        "trainer_backend": str(data.get("trainer_backend", "") or ""),
        "contract_class": str(data.get("contract_class", "") or ""),
        "epoch_based": bool(data.get("epoch_based", False)),
        "promotable": bool(data.get("promotable", False)),
        "resume_capable": bool(data.get("resume_capable", False)),
        "gpu_required": bool(data.get("gpu_required", False)),
        "runtime_env": str(data.get("runtime_env", "") or ""),
        "min_start_epoch_budget": int(data.get("min_start_epoch_budget", 0) or 0),
        "requested_epochs": int(data.get("requested_epochs", 0) or 0),
        "min_epochs": int(data.get("min_epochs", 0) or 0),
        "resume_mode": str(data.get("resume_mode", "") or ""),
    }

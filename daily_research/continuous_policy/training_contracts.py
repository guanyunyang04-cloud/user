from __future__ import annotations

from typing import Any


TRAINER_BACKEND_PROTOTYPE_V1 = "prototype_gbdt_v1"
TRAINER_BACKEND_FORMAL_V2 = "formal_torch_v2"
TRAINER_BACKENDS = (
    TRAINER_BACKEND_PROTOTYPE_V1,
    TRAINER_BACKEND_FORMAL_V2,
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
    if backend == TRAINER_BACKEND_FORMAL_V2:
        requested_epochs = max(int(requested_epochs or 0), 32)
        min_epochs = max(int(min_epochs or 0), 32)
        if not resume_mode_text:
            resume_mode_text = "strict"
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
            "notes": [
                "Formal continuous-policy candidates must run in yolos with CUDA enabled.",
                "Initial formal budget starts at 32 epochs; additional budget must continue via strict resume.",
                "Promotion evaluation is allowed only for this contract class.",
            ],
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

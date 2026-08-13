"""Noms canoniques des artefacts du papier.

Ce registre est la source unique utilisée par les checkpoints, les exports et
les cartes de modèle. Changer un nom ici le change partout.
"""
MODEL_REGISTRY = {
    "A": {
        "slug": "french-serer-nllb-wolof-proximity-probe",
        "display_name": "French–Serer NLLB Wolof-Target Proximity Probe",
        "kind": "evaluation_probe",
        "direction": "French → Wolof (compared with Serer references)",
        "publish_model": False,
    },
    "B": {
        "slug": "french-serer-nllb-full-ft",
        "display_name": "French–Serer NLLB Full Fine-Tuning",
        "kind": "final_translation_model",
        "direction": "French → Serer",
        "publish_model": True,
    },
    "C": {
        "slug": "french-serer-nllb-lora",
        "display_name": "French–Serer NLLB LoRA",
        "kind": "final_translation_model",
        "direction": "French → Serer",
        "publish_model": True,
    },
    "D": {
        "slug": "french-serer-nllb-lora-backtranslation",
        "display_name": "French–Serer NLLB LoRA with Back-Translation",
        "kind": "final_translation_model",
        "direction": "French → Serer",
        "publish_model": True,
    },
    "D_REVERSE": {
        "slug": "serer-french-nllb-full-ft-backtranslator",
        "display_name": "Serer–French NLLB Full-FT Back-Translator",
        "kind": "auxiliary_backtranslation_model",
        "direction": "Serer → French",
        "publish_model": True,
    },
    "E": {
        "slug": "french-serer-opusmt-full-ft",
        "display_name": "French–Serer OPUS-MT Full Fine-Tuning",
        "kind": "final_translation_model",
        "direction": "French → Serer",
        "publish_model": True,
    },
    "F": {
        "slug": "french-serer-transformer-scratch",
        "display_name": "French–Serer Transformer from Scratch",
        "kind": "final_translation_model",
        "direction": "French → Serer",
        "publish_model": True,
    },
    "G": {
        "slug": "french-serer-nllb-lora-srr-token",
        "display_name": "French–Serer NLLB LoRA with Native srr_Latn Token",
        "kind": "final_translation_model",
        "direction": "French → Serer",
        "publish_model": True,
    },
}


def model_metadata(experiment_id: str) -> dict:
    key = experiment_id.upper().replace("-", "_")
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Artefact inconnu: {experiment_id}")
    return {"experiment_id": key, **MODEL_REGISTRY[key]}


def recommended_repo_id(experiment_id: str, organization: str) -> str:
    if not organization or "/" in organization:
        raise ValueError("--organization doit être un nom de compte ou d'organisation HF")
    return f"{organization}/{model_metadata(experiment_id)['slug']}"

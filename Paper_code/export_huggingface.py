"""Exporte les artefacts A–F avec des noms Hugging Face canoniques.

Les dépôts sont privés par défaut. A est une sonde et n'exporte volontairement
aucun poids. E et F incluent leur code d'architecture personnalisé.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from config import (
    BackTranslationConfig, FullFTConfig, LoRAExperimentConfig, OpusMTConfig,
    ProbeConfig, ReverseFullFTConfig, ScratchConfig,
)
from model_registry import model_metadata, recommended_repo_id
from reproducibility import best_checkpoint, write_manifest


ROOT = Path(__file__).resolve().parent


def load(experiment):
    if experiment == "A":
        return ProbeConfig(), None, None, None
    if experiment in ("B", "D_REVERSE"):
        from models.model_full import NLLBFullFineTuner
        if experiment == "B":
            cfg, directory = FullFTConfig(), FullFTConfig().CHECKPOINTS_B
        else:
            cfg = ReverseFullFTConfig()
            directory = Path(cfg.CHECKPOINTS_DIR) / "config_D_reverse"
        checkpoint = best_checkpoint(directory)
        module = NLLBFullFineTuner.load_from_checkpoint(checkpoint, cfg=cfg)
        return cfg, checkpoint, module.model, module.tokenizer
    if experiment in ("C", "D"):
        from models.model_lora import NLLBFineTuner
        cfg = LoRAExperimentConfig() if experiment == "C" else BackTranslationConfig()
        checkpoint = best_checkpoint(cfg.CHECKPOINT_DIR)
        module = NLLBFineTuner.load_from_checkpoint(checkpoint, cfg=cfg)
        return cfg, checkpoint, module.model.merge_and_unload(), module.tokenizer
    if experiment == "E":
        from baselines.models.model_opusmt import OpusMTFineTuner
        cfg = OpusMTConfig()
        checkpoint = best_checkpoint(Path(cfg.CHECKPOINTS_DIR) / "baseline_opusmt")
        module = OpusMTFineTuner.load_from_checkpoint(checkpoint, cfg=cfg)
        return cfg, checkpoint, module.model, module.source_tokenizer
    from baselines.models.model_scratch import ScratchTransformer
    cfg = ScratchConfig()
    checkpoint = best_checkpoint(Path(cfg.CHECKPOINTS_DIR) / "baseline_scratch")
    module = ScratchTransformer.load_from_checkpoint(checkpoint, cfg=cfg)
    return cfg, checkpoint, module.model, module.tokenizer


def write_card(output: Path, metadata: dict, checkpoint: str | None) -> None:
    template = (ROOT / "MODEL_CARD.md").read_text(encoding="utf-8")
    identity = (
        f"\n## Export identity\n\n"
        f"- Canonical name: `{metadata['slug']}`\n"
        f"- Display name: {metadata['display_name']}\n"
        f"- Experiment: {metadata['experiment_id']}\n"
        f"- Artifact type: `{metadata['kind']}`\n"
        f"- Direction: {metadata['direction']}\n"
        f"- Source checkpoint: `{checkpoint or 'none (probe)'}`\n"
    )
    (output / "README.md").write_text(template + identity, encoding="utf-8")


def export_artifact(experiment: str, output: Path):
    metadata = model_metadata(experiment)
    cfg, checkpoint, model, tokenizer = load(experiment)
    output.mkdir(parents=True, exist_ok=True)

    if experiment == "A":
        # Ne jamais republier les 600M poids NLLB sous le nom d'un modèle sérère.
        source = Path(cfg.OUTPUTS_DIR) / "predictions_A.jsonl"
        if source.exists():
            shutil.copy2(source, output / source.name)
    elif experiment == "F":
        from safetensors.torch import save_file
        state = {name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items()}
        save_file(state, str(output / "model.safetensors"))
        shutil.copy2(ROOT / "baselines/tokenizer/spm.model", output / "spm.model")
        shutil.copy2(ROOT / "baselines/models/model_scratch.py", output / "modeling_scratch.py")
        (output / "architecture.json").write_text(json.dumps({
            "vocab_size": tokenizer.vocab_size, "d_model": cfg.D_MODEL, "d_ff": cfg.D_FF,
            "num_heads": cfg.N_HEADS, "encoder_layers": cfg.N_ENCODER_LAYERS,
            "decoder_layers": cfg.N_DECODER_LAYERS, "dropout": cfg.DROPOUT,
        }, indent=2), encoding="utf-8")
    else:
        model.save_pretrained(output, safe_serialization=True)
        tokenizer.save_pretrained(output)
        if experiment == "E":
            shutil.copy2(ROOT / "baselines/tokenizer/spm.model", output / "serer_spm.model")
            shutil.copy2(ROOT / "baselines/models/model_opusmt.py", output / "modeling_opusmt_serer.py")

    complete_metadata = {
        **metadata,
        "checkpoint": checkpoint,
        "base_model": cfg.MODEL_NAME,
        "base_model_revision": cfg.MODEL_REVISION,
        "source_language": "French" if experiment != "D_REVERSE" else "Serer",
        "target_language": "French" if experiment == "D_REVERSE" else ("Wolof probe" if experiment == "A" else "Serer"),
        "nllb_serer_proxy_token": "wol_Latn" if experiment in ("A", "B", "C", "D", "D_REVERSE") else None,
        "uses_proxy_for_serer_side": experiment in ("A", "B", "C", "D", "D_REVERSE"),
    }
    (output / "experiment_metadata.json").write_text(
        json.dumps(complete_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_card(output, metadata, checkpoint)
    write_manifest(cfg, output / "reproducibility_manifest.json", best_checkpoint=checkpoint,
                   canonical_identity=metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["A", "B", "C", "D", "D_REVERSE", "E", "F"], required=True)
    parser.add_argument("--output", help="Dossier local; défaut: hf_export/<nom-canonique>")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--organization", help="Compte/organisation HF; génère automatiquement le repo-id canonique")
    parser.add_argument("--public", action="store_true", help="Rend explicitement le dépôt HF public")
    args = parser.parse_args()
    metadata = model_metadata(args.experiment)
    output = Path(args.output) if args.output else ROOT / "hf_export" / metadata["slug"]
    export_artifact(args.experiment, output)
    print(f"Export local: {output}")
    if args.push:
        if not args.organization:
            parser.error("--organization est obligatoire avec --push")
        repo_id = recommended_repo_id(args.experiment, args.organization)
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(repo_id, private=not args.public, exist_ok=True)
        api.upload_folder(repo_id=repo_id, folder_path=str(output))
        print(f"Dépôt Hugging Face: {repo_id} ({'public' if args.public else 'privé'})")


if __name__ == "__main__":
    main()

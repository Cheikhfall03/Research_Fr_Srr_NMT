"""Interface d'inférence unique pour tous les artefacts A–F et D_REVERSE.

Exemples:
  python inference/translate.py --experiment C --text "Bonjour"
  python inference/translate.py --experiment D_REVERSE --text "Phrase sérère"
  python inference/translate.py --experiment E --input phrases.txt --output predictions.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForSeq2SeqLM, NllbTokenizerFast

from config import (
    BackTranslationConfig, FullFTConfig, LoRAExperimentConfig, OpusMTConfig,
    ProbeConfig, ReverseFullFTConfig, ScratchConfig,
)
from model_registry import model_metadata
from reproducibility import best_checkpoint


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_artifact(experiment: str):
    """Retourne configuration, modèle, tokenizers et checkpoint."""
    if experiment == "A":
        cfg = ProbeConfig()
        tokenizer = NllbTokenizerFast.from_pretrained(
            cfg.MODEL_NAME, revision=cfg.MODEL_REVISION,
            src_lang=cfg.SRC_LANG, tgt_lang=cfg.TGT_LANG,
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            cfg.MODEL_NAME, revision=cfg.MODEL_REVISION
        )
        return cfg, model, tokenizer, tokenizer, None

    if experiment in ("B", "D_REVERSE"):
        from models.model_full import NLLBFullFineTuner
        if experiment == "B":
            cfg, directory = FullFTConfig(), FullFTConfig().CHECKPOINTS_B
        else:
            cfg = ReverseFullFTConfig()
            directory = Path(cfg.CHECKPOINTS_DIR) / "config_D_reverse"
        checkpoint = best_checkpoint(directory)
        module = NLLBFullFineTuner.load_from_checkpoint(checkpoint, cfg=cfg)
        return cfg, module.model, module.tokenizer, module.tokenizer, checkpoint

    if experiment in ("C", "D"):
        from models.model_lora import NLLBFineTuner
        cfg = LoRAExperimentConfig() if experiment == "C" else BackTranslationConfig()
        checkpoint = best_checkpoint(cfg.CHECKPOINT_DIR)
        module = NLLBFineTuner.load_from_checkpoint(checkpoint, cfg=cfg)
        return cfg, module.model, module.tokenizer, module.tokenizer, checkpoint

    if experiment == "E":
        from baselines.models.model_opusmt import OpusMTFineTuner
        cfg = OpusMTConfig()
        checkpoint = best_checkpoint(Path(cfg.CHECKPOINTS_DIR) / "baseline_opusmt")
        module = OpusMTFineTuner.load_from_checkpoint(checkpoint, cfg=cfg)
        return cfg, module.model, module.source_tokenizer, module.target_tokenizer, checkpoint

    from baselines.models.model_scratch import ScratchTransformer
    cfg = ScratchConfig()
    checkpoint = best_checkpoint(Path(cfg.CHECKPOINTS_DIR) / "baseline_scratch")
    module = ScratchTransformer.load_from_checkpoint(checkpoint, cfg=cfg)
    return cfg, module.model, module.tokenizer, module.tokenizer, checkpoint


@torch.inference_mode()
def translate_batch(experiment, cfg, model, source_tokenizer, target_tokenizer, texts):
    model.eval().to(DEVICE)
    if experiment == "F":
        encoded = [source_tokenizer.encode(text, cfg.MAX_LENGTH) for text in texts]
        padded, masks = source_tokenizer.pad(encoded, cfg.MAX_LENGTH)
        source = torch.tensor(padded, dtype=torch.long, device=DEVICE)
        attention = torch.tensor(masks, dtype=torch.long, device=DEVICE)
        generated = model.generate(source, attention, max_new_tokens=cfg.MAX_LENGTH)
        return [target_tokenizer.decode(row.tolist()) for row in generated]

    inputs = source_tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True,
        max_length=cfg.MAX_LENGTH,
    ).to(DEVICE)
    kwargs = {"max_new_tokens": cfg.MAX_LENGTH, "num_beams": cfg.NUM_BEAMS_TEST}
    if experiment in ("A", "B", "C", "D", "D_REVERSE"):
        kwargs["forced_bos_token_id"] = source_tokenizer.convert_tokens_to_ids(cfg.TGT_LANG)
    generated = model.generate(**inputs, **kwargs)
    if experiment == "E":
        return target_tokenizer.batch_decode(generated)
    return target_tokenizer.batch_decode(generated, skip_special_tokens=True)


def read_inputs(text: str | None, input_path: str | None) -> list[str]:
    if text:
        return [text]
    path = Path(input_path)
    rows = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"Aucune phrase dans {path}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["A", "B", "C", "D", "D_REVERSE", "E", "F"], required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="Une phrase source")
    source.add_argument("--input", help="Fichier UTF-8, une phrase par ligne")
    parser.add_argument("--output", help="Sortie JSONL; affichage console si omis")
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    identity = model_metadata(args.experiment)
    if args.experiment == "A":
        print("AVERTISSEMENT: A produit du wolof; ce n'est pas un traducteur français→sérère.", file=sys.stderr)
    texts = read_inputs(args.text, args.input)
    cfg, model, source_tokenizer, target_tokenizer, checkpoint = load_artifact(args.experiment)
    batch_size = args.batch_size or cfg.BATCH_SIZE
    predictions = []
    for start in range(0, len(texts), batch_size):
        predictions.extend(translate_batch(
            args.experiment, cfg, model, source_tokenizer, target_tokenizer,
            texts[start:start + batch_size],
        ))
    records = [{
        "experiment_id": identity["experiment_id"], "model_name": identity["slug"],
        "direction": identity["direction"], "checkpoint": checkpoint,
        "source": source, "prediction": prediction,
    } for source, prediction in zip(texts, predictions)]
    rendered = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records)
    if args.output:
        output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
        print(f"{len(records)} prédiction(s) → {output}")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()

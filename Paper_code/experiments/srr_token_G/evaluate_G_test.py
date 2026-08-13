"""Évaluation finale de l'expérience G sur le vrai test set (beam=5).

Le val_BLEU produit par train_G_srr_token.py (beam=1, split val, calculé
pendant l'entraînement pour la sélection de checkpoint) n'est PAS comparable
aux chiffres du papier ni à ceux des configs A-F évaluées par
evaluate_test.py — celui-ci utilise le protocole final identique (beam=5,
split test) pour produire un résultat réellement comparable.

Réutilise generate()/metrics() d'evaluate_test.py pour garantir un protocole
strictement identique aux autres configs.

Usage:
    python experiments/srr_token_G/evaluate_G_test.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent
ROOT = EXPERIMENT_DIR.parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from reproducibility import best_checkpoint, sha256, write_manifest  # noqa: E402
from evaluate_test import DEVICE, generate, metrics  # noqa: E402
from models.model_lora import NLLBFineTuner  # noqa: E402
from experiments.srr_token_G.train_G_srr_token import SrrTokenConfig  # noqa: E402


def main() -> None:
    cfg = SrrTokenConfig()
    checkpoint = best_checkpoint(cfg.CHECKPOINTS_G)
    print(f"Checkpoint -> {checkpoint}")

    module = NLLBFineTuner.load_from_checkpoint(checkpoint, cfg=cfg)
    model, tokenizer = module.model, module.tokenizer
    model.eval().to(DEVICE)

    test_path = Path(cfg.DATA_DIR) / "test.json"
    rows = json.loads(test_path.read_text(encoding="utf-8"))
    refs = [x["serere"] for x in rows]

    predictions, loss = generate(model, tokenizer, rows, cfg)
    score = {**metrics(predictions, refs, cfg), "loss": loss,
             "checkpoint": checkpoint, "test_sha256": sha256(test_path), "seed": cfg.SEED}

    prediction_rows = [{"id": i, "francais": row["francais"], "serere_reference": row["serere"],
                        "prediction": pred} for i, (row, pred) in enumerate(zip(rows, predictions))]
    (EXPERIMENT_DIR / "outputs").mkdir(parents=True, exist_ok=True)
    (EXPERIMENT_DIR / "outputs" / "predictions_G.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in prediction_rows), encoding="utf-8")

    (EXPERIMENT_DIR / "results").mkdir(parents=True, exist_ok=True)
    (EXPERIMENT_DIR / "results" / "evaluation_test_G.json").write_text(
        json.dumps({"G": score}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_manifest(cfg, EXPERIMENT_DIR / "results" / "evaluation_G_manifest.json", [test_path], metrics=score)

    print(json.dumps(score, ensure_ascii=False, indent=2))
    print(f"\nComparez a results/evaluation_test.json (configs A-D, meme protocole beam=5) "
          f"pour situer G face a C ({cfg.TGT_LANG} natif vs wol_Latn proxy).")


if __name__ == "__main__":
    main()

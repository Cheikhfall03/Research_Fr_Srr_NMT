"""Calibration du proximity probe (et des systèmes adaptés) sur un domaine non-religieux.

Répond à la demande du Reviewer 3: le corpus principal est ~90% biblique, ce qui
pourrait amplifier artificiellement le chevauchement lexical français-wolof-sérère
(emprunts religieux d'origine arabe). Ce script rejoue l'évaluation sur
data/udhr_fr_srr_probe_test.json (50 phrases de la Déclaration universelle des
droits de l'homme, domaine juridique/institutionnel, hors Bible) afin de vérifier
si le score "wol_Latn" du probe (config A) et le comportement des systèmes adaptés
se maintiennent hors du registre religieux — ce qui renforcerait l'hypothèse de
proximité phylogénétique Wolof-Sérère plutôt que celle d'un simple effet de domaine.

Usage:
    python evaluate_udhr_calibration.py --configs A B C D
    python evaluate_udhr_calibration.py --configs A --seed 43
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from config import ProbeConfig
from evaluate_test import DEVICE, generate, load_experiment, metrics
from reproducibility import sha256, write_manifest

import torch

UDHR_PATH_NAME = "udhr_fr_srr_probe_test.json"
CALIBRATION_DIR = "results/calibration_udhr"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # E/F (baselines opus-mt / scratch) utilisent un chargement de checkpoint différent,
    # non couvert par load_experiment() (réutilisé d'evaluate_test.py) — non supportés ici.
    parser.add_argument("--configs", nargs="+", choices=list("ABCD"), default=["A"])
    parser.add_argument("--seed", type=int, default=None,
                         help="Charge le checkpoint entraîné avec ce seed (config_X_seedN). Ignoré pour A.")
    args = parser.parse_args()
    suffix = "" if args.seed is None else f"_seed{args.seed}"

    base = ProbeConfig()
    udhr_path = Path(base.DATA_DIR) / UDHR_PATH_NAME
    rows = json.loads(udhr_path.read_text(encoding="utf-8"))
    refs = [x["serere"] for x in rows]

    calibration_dir = Path(CALIBRATION_DIR)
    calibration_dir.mkdir(parents=True, exist_ok=True)

    all_scores = {}
    for name in args.configs:
        cfg, model, tokenizer, checkpoint = load_experiment(name, seed=args.seed)
        model.eval().to(DEVICE)
        predictions, loss = generate(model, tokenizer, rows, cfg)
        score = {**metrics(predictions, refs, cfg), "loss": loss, "checkpoint": checkpoint,
                 "udhr_sha256": sha256(udhr_path), "n_sentences": len(rows), "domain": "udhr_non_religious"}
        all_scores[name] = score

        prediction_rows = [{"id": i, "francais": row["francais"], "serere_reference": row["serere"],
                            "prediction": pred} for i, (row, pred) in enumerate(zip(rows, predictions))]
        (calibration_dir / f"predictions_{name}{suffix}.jsonl").write_text(
            "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in prediction_rows), encoding="utf-8")
        write_manifest(cfg, calibration_dir / f"calibration_{name}{suffix}_manifest.json", [udhr_path], metrics=score)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    (calibration_dir / f"calibration_udhr{suffix}.json").write_text(
        json.dumps(all_scores, ensure_ascii=False, indent=2), encoding="utf-8")

    fields = ["config", "bleu", "chrf", "rouge1", "rougeL", "bertscore_f1", "loss", "n_sentences", "checkpoint"]
    with open(calibration_dir / f"calibration_udhr{suffix}.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, score in all_scores.items():
            writer.writerow({"config": name, **{k: score[k] for k in fields if k != "config"}})

    print(f"\n=== Calibration UDHR (domaine non-religieux, n={len(rows)}) ===")
    for name, score in all_scores.items():
        print(f"Config {name}: BLEU={score['bleu']}  chrF={score['chrf']}  BERTScore-F1={score['bertscore_f1']}")
    print(f"\nComparez à results/evaluation_test{suffix}.json (corpus principal, ~90% biblique) "
          "pour évaluer si le score du probe/des systèmes se maintient hors du domaine religieux.")
    print(f"Résultats -> {calibration_dir}")


if __name__ == "__main__":
    main()

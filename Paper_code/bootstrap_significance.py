"""Test de significativité par bootstrap resampling apparié (Koehn, 2004).

Complète (sans la remplacer) l'étude multi-seed: ce test rééchantillonne le
test set à sorties de modèle fixes, donc il quantifie le bruit d'échantillonnage
du test set, pas la variance d'initialisation/entraînement (voir
training/run_multiseed.py pour cette dernière).

Prérequis: avoir déjà lancé evaluate_test.py, qui écrit
outputs/predictions_<CONFIG>.jsonl (ou predictions_<CONFIG>_seedN.jsonl).

Usage:
    python bootstrap_significance.py C D
    python bootstrap_significance.py C D --n-resamples 2000
    python bootstrap_significance.py C D --suffix _seed42
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import sacrebleu


def load_predictions(path: Path) -> tuple[list[str], list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"{path} introuvable — lancez d'abord evaluate_test.py.")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    predictions = [r["prediction"] for r in rows]
    references = [r["serere_reference"] for r in rows]
    return predictions, references


def paired_bootstrap(preds_a, preds_b, refs, n_resamples=1000, seed=42):
    n = len(refs)
    if len(preds_a) != n or len(preds_b) != n:
        raise ValueError("Les deux systèmes doivent partager le même test set, dans le même ordre.")
    rng = np.random.default_rng(seed)

    bleu_a = sacrebleu.corpus_bleu(preds_a, [refs]).score
    bleu_b = sacrebleu.corpus_bleu(preds_b, [refs]).score
    observed_delta = bleu_a - bleu_b

    deltas = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        sample_refs = [refs[j] for j in idx]
        sample_a = [preds_a[j] for j in idx]
        sample_b = [preds_b[j] for j in idx]
        score_a = sacrebleu.corpus_bleu(sample_a, [sample_refs]).score
        score_b = sacrebleu.corpus_bleu(sample_b, [sample_refs]).score
        deltas[i] = score_a - score_b

    # p bilatéral: fréquence à laquelle le rééchantillonnage inverse le signe de l'écart observé.
    p_value = float(np.mean(deltas <= 0)) if observed_delta >= 0 else float(np.mean(deltas >= 0))
    ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])

    return {
        "bleu_a": round(bleu_a, 4),
        "bleu_b": round(bleu_b, 4),
        "observed_delta": round(observed_delta, 4),
        "delta_ci95_low": round(float(ci_low), 4),
        "delta_ci95_high": round(float(ci_high), 4),
        "p_value": round(p_value, 5),
        "significant_at_0.05": bool(p_value < 0.05),
        "n_resamples": n_resamples,
        "n_sentences": n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("system_a", help="Lettre de config du système A, ex. C")
    parser.add_argument("system_b", help="Lettre de config du système B, ex. D")
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--suffix", default="", help="Suffixe de fichier, ex. _seed42")
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42, help="Seed du générateur aléatoire du bootstrap lui-même.")
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    preds_a, refs_a = load_predictions(outputs_dir / f"predictions_{args.system_a}{args.suffix}.jsonl")
    preds_b, refs_b = load_predictions(outputs_dir / f"predictions_{args.system_b}{args.suffix}.jsonl")
    if refs_a != refs_b:
        raise ValueError("Les deux systèmes doivent partager le même test set, dans le même ordre.")

    result = paired_bootstrap(preds_a, preds_b, refs_a, n_resamples=args.n_resamples, seed=args.seed)
    result["system_a"] = args.system_a
    result["system_b"] = args.system_b
    print(json.dumps(result, ensure_ascii=False, indent=2))

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"bootstrap_{args.system_a}_vs_{args.system_b}{args.suffix}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nÉcrit dans {out_path}")


if __name__ == "__main__":
    main()

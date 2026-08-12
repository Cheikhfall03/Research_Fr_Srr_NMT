"""Statistiques de longueur des phrases + exemples du format d'entrée du modèle.

Répond à la remarque du Reviewer 1: les traductions sont évaluées phrase à
phrase, sans contexte au-delà de la phrase, et l'article ne précise ni les
longueurs typiques ni le format exact envoyé au modèle. Ce script:

1. calcule les statistiques de longueur (mots et tokens NLLB) par split
   (train/val/test) et pour le set de calibration UDHR;
2. sérialise 3 exemples réels montrant l'entrée telle qu'elle est réellement
   soumise au modèle (texte source, tag de langue source, tag de langue cible
   forcé en décodage) — NLLB ne prend pas de "prompt" textuel libre comme un
   LLM généraliste: le contexte est entièrement porté par les tokens spéciaux
   src_lang/tgt_lang, ce que ce script rend explicite pour la section
   méthodologie de l'article.

Usage:
    python corpus_statistics.py
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

from transformers import NllbTokenizerFast

from config import ProbeConfig

OUTPUT_DIR = Path("results/corpus_statistics")


def word_count(text: str) -> int:
    return len(text.split())


def length_stats(rows: list[dict], tokenizer: NllbTokenizerFast) -> dict:
    fr_words = [word_count(r["francais"]) for r in rows]
    sr_words = [word_count(r["serere"]) for r in rows]
    fr_tokens = [len(tokenizer(r["francais"])["input_ids"]) for r in rows]
    sr_tokens = [len(tokenizer(r["serere"])["input_ids"]) for r in rows]

    def summarize(values: list[int]) -> dict:
        return {
            "min": min(values), "max": max(values),
            "mean": round(st.mean(values), 2), "median": st.median(values),
            "p95": round(sorted(values)[int(0.95 * (len(values) - 1))], 2),
        }

    return {
        "n_sentences": len(rows),
        "francais_words": summarize(fr_words),
        "serere_words": summarize(sr_words),
        "francais_nllb_tokens": summarize(fr_tokens),
        "serere_nllb_tokens": summarize(sr_tokens),
    }


def example_model_inputs(rows: list[dict], tokenizer: NllbTokenizerFast, cfg, n: int = 3) -> list[dict]:
    target_id = tokenizer.convert_tokens_to_ids(cfg.TGT_LANG)
    examples = []
    for row in rows[:n]:
        encoded = tokenizer(row["francais"], text_target=row["serere"], truncation=True,
                            max_length=cfg.MAX_LENGTH)
        examples.append({
            "francais_source": row["francais"],
            "serere_reference": row["serere"],
            "source_lang_tag": cfg.SRC_LANG,
            "target_lang_tag_decoded_with": cfg.TGT_LANG,
            "forced_bos_token_id": target_id,
            "source_input_ids": encoded["input_ids"],
            "source_tokens": tokenizer.convert_ids_to_tokens(encoded["input_ids"]),
            "max_length": cfg.MAX_LENGTH,
            "num_beams_inference": cfg.NUM_BEAMS_TEST,
            "note": "NLLB ne reçoit pas de prompt en langage naturel: le contexte de traduction est "
                    "porté par le tag de langue source préfixé au texte et par forced_bos_token_id "
                    "qui impose la langue cible au décodeur — il n'y a pas de contexte inter-phrase, "
                    "chaque phrase est traduite indépendamment.",
        })
    return examples


def main() -> None:
    cfg = ProbeConfig()
    tokenizer = NllbTokenizerFast.from_pretrained(cfg.MODEL_NAME, revision=cfg.MODEL_REVISION,
                                                   src_lang=cfg.SRC_LANG, tgt_lang=cfg.TGT_LANG)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stats = {}
    for split in ("train", "val", "test"):
        path = Path(cfg.DATA_DIR) / f"{split}.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        stats[split] = length_stats(rows, tokenizer)

    udhr_path = Path(cfg.DATA_DIR) / "udhr_fr_srr_probe_test.json"
    if udhr_path.exists():
        udhr_rows = json.loads(udhr_path.read_text(encoding="utf-8"))
        stats["udhr_calibration"] = length_stats(udhr_rows, tokenizer)

    (OUTPUT_DIR / "length_statistics.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    test_rows = json.loads((Path(cfg.DATA_DIR) / "test.json").read_text(encoding="utf-8"))
    examples = example_model_inputs(test_rows, tokenizer, cfg, n=3)
    (OUTPUT_DIR / "example_model_inputs.json").write_text(
        json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Longueurs par split (mots) ===")
    for split, values in stats.items():
        fr, sr = values["francais_words"], values["serere_words"]
        print(f"{split}: n={values['n_sentences']}  FR words mean={fr['mean']} (median={fr['median']}, "
              f"p95={fr['p95']}, max={fr['max']})  SRR words mean={sr['mean']} (median={sr['median']}, "
              f"p95={sr['p95']}, max={sr['max']})")
    print(f"\nStatistiques complètes -> {OUTPUT_DIR / 'length_statistics.json'}")
    print(f"Exemples de format d'entrée modèle -> {OUTPUT_DIR / 'example_model_inputs.json'}")


if __name__ == "__main__":
    main()

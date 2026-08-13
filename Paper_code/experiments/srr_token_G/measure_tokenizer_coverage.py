"""Mesure la couverture du tokenizer NLLB existant sur le corpus sérère.

À lancer AVANT d'envisager une extension de vocabulaire (voir README.md,
étape 1). Calcule, pour le français (référence, langue bien couverte par
NLLB) et le sérère, deux indicateurs sur data/train.json:

- tokens/mot: combien de sous-mots en moyenne pour écrire un mot. Un ratio
  nettement plus élevé pour le sérère que pour le français indique une
  couverture pauvre (le tokenizer découpe en fragments minuscules faute de
  connaître les motifs de lettres du sérère).
- taux de <unk>: proportion de tokens remplacés par l'inconnu — perte
  d'information totale sur ces caractères. Signal le plus grave; s'il est
  non nul, une extension de vocabulaire devient nécessaire, pas seulement
  optionnelle.

Usage:
    python experiments/srr_token_G/measure_tokenizer_coverage.py
"""
from __future__ import annotations

import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

from transformers import NllbTokenizerFast

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import ProbeConfig  # noqa: E402

OUTPUT_PATH = Path(__file__).resolve().parent / "coverage_report.json"

# Ponctuation typographique dont l'absence du vocabulaire NLLB n'est pas
# spécifique à une langue (observé aussi sur le français) — normalisée avant
# mesure pour isoler le vrai signal de couverture propre au sérère.
PUNCTUATION_NORMALIZATION = {
    "«": '"', "»": '"', "“": '"', "”": '"',
    "‘": "'", "’": "'", "–": "-", "—": "-",
}


def normalize_punctuation(text: str) -> str:
    for old, new in PUNCTUATION_NORMALIZATION.items():
        text = text.replace(old, new)
    return unicodedata.normalize("NFC", text)


def measure(tokenizer: NllbTokenizerFast, texts: list[str], lang_tag: str, normalize: bool) -> dict:
    tokenizer.src_lang = lang_tag
    total_words, total_tokens, total_unk = 0, 0, 0
    unk_id = tokenizer.unk_token_id
    unk_chars: Counter[str] = Counter()
    for raw_text in texts:
        text = normalize_punctuation(raw_text) if normalize else raw_text
        words = text.split()
        total_words += len(words)
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        for token_id, offset in zip(encoded["input_ids"], encoded["offset_mapping"]):
            total_tokens += 1
            if token_id == unk_id:
                total_unk += 1
                if offset:
                    unk_chars[text[offset[0]:offset[1]]] += 1
    return {
        "n_sentences": len(texts),
        "n_words": total_words,
        "n_tokens": total_tokens,
        "tokens_per_word": round(total_tokens / max(total_words, 1), 3),
        "unk_count": total_unk,
        "unk_rate_pct": round(100 * total_unk / max(total_tokens, 1), 4),
        "unk_characters_top10": unk_chars.most_common(10),
    }


def main() -> None:
    cfg = ProbeConfig()
    tokenizer = NllbTokenizerFast.from_pretrained(cfg.MODEL_NAME, revision=cfg.MODEL_REVISION)

    rows = json.loads((Path(cfg.DATA_DIR) / "train.json").read_text(encoding="utf-8"))
    french = [r["francais"] for r in rows]
    serer = [r["serere"] for r in rows]

    report = {
        "francais_brut": measure(tokenizer, french, "fra_Latn", normalize=False),
        "serere_brut": measure(tokenizer, serer, "wol_Latn", normalize=False),
        "francais_normalise": measure(tokenizer, french, "fra_Latn", normalize=True),
        "serere_normalise": measure(tokenizer, serer, "wol_Latn", normalize=True),
    }

    fr_tpw = report["francais_normalise"]["tokens_per_word"]
    sr_tpw = report["serere_normalise"]["tokens_per_word"]
    fr_unk = report["francais_normalise"]["unk_rate_pct"]
    sr_unk = report["serere_normalise"]["unk_rate_pct"]
    residual_unk = sr_unk - fr_unk  # au-delà du bruit de ponctuation partagé avec le français

    if residual_unk > 0.05:
        verdict = (f"EXTENSION CIBLÉE NÉCESSAIRE: après normalisation de la ponctuation, "
                   f"le sérère garde {residual_unk:.3f} points de <unk> de plus que le français "
                   f"(voir unk_characters_top10 de serere_normalise) -> ce sont des lettres "
                   f"sérères réellement absentes du vocabulaire, pas de la ponctuation -> "
                   f"ajouter ces quelques caractères manquants au vocabulaire (extension "
                   f"minimale, pas un nouveau tokenizer complet).")
    elif sr_tpw > 1.5 * fr_tpw:
        verdict = ("MOYEN: après normalisation, le sérère reste bien plus fragmenté que le "
                   "français (ratio > 1.5x) -> extension de vocabulaire probablement utile.")
    else:
        verdict = ("BON après normalisation de la ponctuation: couverture comparable au "
                   "français -> extension de vocabulaire inutile au-delà des quelques "
                   "caractères résiduels listés, le token de langue seul (srr_Latn, voir "
                   "add_srr_token.py) devrait suffire.")

    report["verdict"] = verdict
    OUTPUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n{verdict}")
    print(f"\nRapport -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

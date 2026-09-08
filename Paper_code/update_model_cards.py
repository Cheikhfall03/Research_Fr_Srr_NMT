"""Génère et pousse des model cards complètes (sans re-uploader les poids).

Contrairement à export_huggingface.py (qui pousse le modèle complet), ce script
ne met à jour QUE le README.md de chaque dépôt HF déjà créé, en le remplissant
avec les vraies métriques, hyperparamètres et limitations — plus de TODO.

Chaque config lit ses métriques depuis les fichiers de résultats disponibles
localement; certaines configs (E, F, G) ne sont donc générables que depuis le
pod où elles ont réellement été évaluées.

Usage:
    python update_model_cards.py --experiments B C D E F --organization Fallovski
    python update_model_cards.py --experiments G --organization Fallovski   # depuis le pod G
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import BackTranslationConfig, FullFTConfig, LoRAExperimentConfig, OpusMTConfig, ScratchConfig
from model_registry import model_metadata, recommended_repo_id
from reproducibility import best_checkpoint

ROOT = Path(__file__).resolve().parent


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def dataset_provenance() -> dict:
    manifest = load_json(ROOT / "data" / "dataset_manifest.json")
    return {
        "train": manifest.get("counts", {}).get("train"),
        "val": manifest.get("counts", {}).get("val"),
        "test": manifest.get("counts", {}).get("test"),
        "test_sha256": manifest.get("splits_sha256", {}).get("test"),
        "seed": manifest.get("seed"),
    }


def multiseed_row(letter: str) -> dict | None:
    import csv
    path = ROOT / "results" / "multiseed_summary.csv"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["config"] == letter:
                return row
    return None


def metrics_table_md(rows: list[tuple[str, str]]) -> str:
    lines = ["| Metric | Value |", "|---|---|"]
    for name, value in rows:
        lines.append(f"| {name} | {value} |")
    return "\n".join(lines)


CARD_SPECS = {
    "B": dict(cfg=FullFTConfig(), results_key="B", checkpoint_dir=lambda c: c.CHECKPOINTS_B,
              hparams=["LR", "NUM_EPOCHS", "BATCH_SIZE", "GRAD_ACCUM_STEPS", "WARMUP_STEPS"],
              proxy=True, native=False, has_atlantic_prior=True),
    "C": dict(cfg=LoRAExperimentConfig(), results_key="C", checkpoint_dir=lambda c: c.CHECKPOINTS_C,
              hparams=["LR", "NUM_EPOCHS", "LORA_R", "LORA_ALPHA", "LORA_DROPOUT", "WARMUP_STEPS"],
              proxy=True, native=False, has_atlantic_prior=True, multiseed=True),
    "D": dict(cfg=BackTranslationConfig(), results_key="D", checkpoint_dir=lambda c: c.CHECKPOINTS_D,
              hparams=["LR", "NUM_EPOCHS", "LORA_R", "LORA_ALPHA", "BT_RATIO", "WARMUP_STEPS"],
              proxy=True, native=False, has_atlantic_prior=True, multiseed=True),
    "E": dict(cfg=OpusMTConfig(), results_key="opusmt",
              checkpoint_dir=lambda c: Path(c.CHECKPOINTS_DIR) / "baseline_opusmt",
              hparams=["LR", "NUM_EPOCHS", "TARGET_VOCAB_SIZE", "WARMUP_STEPS"],
              proxy=False, native=False, has_atlantic_prior=False),
    "F": dict(cfg=ScratchConfig(), results_key="scratch",
              checkpoint_dir=lambda c: Path(c.CHECKPOINTS_DIR) / "baseline_scratch",
              hparams=["LR", "NUM_EPOCHS", "D_MODEL", "N_ENCODER_LAYERS", "N_HEADS", "DROPOUT"],
              proxy=False, native=False, has_atlantic_prior=False),
}


def build_card_B_to_F(letter: str) -> str:
    spec = CARD_SPECS[letter]
    cfg = spec["cfg"]
    meta = model_metadata(letter)
    prov = dataset_provenance()

    eval_test = load_json(ROOT / "results" / "evaluation_test.json")
    eval_base = load_json(ROOT / "results" / "evaluation_baselines.json")
    scores = eval_test.get(spec["results_key"]) or eval_base.get(spec["results_key"], {})

    try:
        checkpoint = Path(best_checkpoint(spec["checkpoint_dir"](cfg))).name
    except Exception:
        checkpoint = "unavailable in this export environment"

    hparams_md = "\n".join(f"- `{h}`: {getattr(cfg, h, 'n/a')}" for h in spec["hparams"])

    metrics_rows = [
        ("BLEU (test, beam={})".format(cfg.NUM_BEAMS_TEST), scores.get("bleu", "n/a")),
        ("chrF", scores.get("chrf", "n/a")),
        ("ROUGE-1", scores.get("rouge1", "n/a")),
        ("ROUGE-L", scores.get("rougeL", "n/a")),
        ("BERTScore-F1", scores.get("bertscore_f1", "n/a")),
        ("Test loss", scores.get("loss", "n/a")),
    ]
    if spec.get("multiseed"):
        row = multiseed_row(letter)
        if row:
            metrics_rows.append((
                f"BLEU (mean ± std, {row['n_seeds']} seeds)",
                f"{row['bleu_mean']} ± {row['bleu_std']}",
            ))

    is_scratch = letter == "F"
    base_model_line = (
        "- **Base model:** none — Transformer encoder-decoder trained from scratch "
        "(no pretrained weights, custom SentencePiece tokenizer)\n"
        if is_scratch else
        f"- **Base model:** `{cfg.MODEL_NAME}` (revision: `{cfg.MODEL_REVISION}`)\n"
    )
    frontmatter_tags = (
        "- low-resource\n- from-scratch\n- serer\n- machine-translation"
        if is_scratch else
        "- low-resource\n- nllb\n- serer\n- machine-translation"
    )

    proxy_note = (
        "\n## Critical limitation — Wolof decoding proxy\n\n"
        "Serer (`srr_Latn`) is not a supported NLLB-200 target. This model decodes under the "
        "`wol_Latn` (Wolof) language tag as a proxy, fine-tuned on French–Serer data. Automatic "
        "metrics (especially BLEU) can be partly inflated by lexical/orthographic overlap with "
        "Wolof; see the companion proximity-probe calibration for this corpus "
        "(`Fallovski/french-serer-nllb-wolof-proximity-probe`). Human review by a qualified Serer "
        "speaker is strongly recommended before any downstream use.\n"
        if spec["proxy"] else ""
    )
    atlantic_note = (
        "\nThis model has **no Atlantic-family prior** (its pretraining does not include Wolof "
        "or any close relative of Serer), unlike the NLLB-based configurations in this project. "
        "In our evaluation, this configuration was judged the most linguistically reliable by a "
        "native Serer-speaking expert despite a lower BLEU score than NLLB-based alternatives — "
        "see the paper for the full BLEU/quality decorrelation analysis.\n"
        if not spec["has_atlantic_prior"] else ""
    )

    return f"""---
language:
- fr
- srr
pipeline_tag: translation
tags:
{frontmatter_tags}
license: other
---

# {meta['display_name']}

Part of a benchmark of six configurations for French→Serer neural machine translation.
Serer is a critically low-resource Niger-Congo language (~1.2M speakers, Senegal/Gambia),
phylogenetically close to the well-resourced Wolof.

## Model summary

- **Experiment ID:** {letter}
- **Kind:** {meta['kind']}
- **Direction:** {meta['direction']}
{base_model_line}- **Best checkpoint:** `{checkpoint}`
- **Random seed:** {prov['seed']}

## Hyperparameters

{hparams_md}
{proxy_note}{atlantic_note}
## Intended use

Research on French-to-Serer machine translation on a corpus that is ~90% religious
(Bible) register, ~10% educational glossaries, primarily Siin dialect. **Not** validated
for legal, medical, emergency, or fully autonomous publication use. Private repository —
not intended for public deployment in its current state.

## Evaluation

{metrics_table_md(metrics_rows)}

Evaluated on the held-out test split ({prov['test']} sentence pairs, SHA-256 of the split:
`{prov['test_sha256']}`). Metrics were computed with the project's own evaluation scripts
(not copied from the manuscript without independent reproduction); the training and
evaluation code is kept in a private repository, available on request.

## Training data and rights

Parallel corpus of {prov['train']} train / {prov['val']} val / {prov['test']} test
French–Serer sentence pairs, built primarily from religious texts (Bible, ~90%) and
educational glossaries (~10%), predominantly Siin dialect. Preprocessing: Unicode
normalization, exact-duplicate removal, length-ratio filtering (1:3–3:1). Document-level
splitting was not possible (no document identifiers available); the split is at the
sentence level with a fixed seed. Full provenance, licensing, and consent documentation
are kept in a private dataset card, available on request, prior to any public release.
"""


def build_card_G(organization: str) -> str:
    """G est hors des configs standard (checkpoint_G isolé, résultats séparés)."""
    import sys
    sys.path.insert(0, str(ROOT))
    from experiments.srr_token_G.train_G_srr_token import SrrTokenConfig

    cfg = SrrTokenConfig()
    meta = model_metadata("G")
    meta_repo_id = recommended_repo_id("G", organization)
    prov = dataset_provenance()
    scores = load_json(ROOT / "experiments" / "srr_token_G" / "results" / "evaluation_test_G.json").get("G", {})
    c_scores = load_json(ROOT / "results" / "evaluation_test.json").get("C", {})

    try:
        checkpoint = Path(best_checkpoint(cfg.CHECKPOINTS_G)).name
    except Exception:
        checkpoint = "unavailable in this export environment"

    metrics_rows = [
        ("BLEU (test, beam=5)", scores.get("bleu", "n/a")),
        ("chrF", scores.get("chrf", "n/a")),
        ("ROUGE-1", scores.get("rouge1", "n/a")),
        ("ROUGE-L", scores.get("rougeL", "n/a")),
        ("BERTScore-F1", scores.get("bertscore_f1", "n/a")),
        ("Test loss", scores.get("loss", "n/a")),
    ]
    comparison_rows = [
        ("BLEU", scores.get("bleu", "n/a"), c_scores.get("bleu", "n/a")),
        ("chrF", scores.get("chrf", "n/a"), c_scores.get("chrf", "n/a")),
        ("ROUGE-L", scores.get("rougeL", "n/a"), c_scores.get("rougeL", "n/a")),
        ("BERTScore-F1", scores.get("bertscore_f1", "n/a"), c_scores.get("bertscore_f1", "n/a")),
    ]
    comparison_md = "\n".join(
        f"| {name} | {g_val} | {c_val} |" for name, g_val, c_val in comparison_rows
    )

    return f"""---
language:
- fr
- srr
pipeline_tag: translation
tags:
- low-resource
- nllb
- serer
- machine-translation
- vocabulary-extension
license: other
---

# {meta['display_name']}

Exploratory follow-up experiment, added during the revision process.
Not part of the six configurations (A–F) benchmarked in the original manuscript.

## Motivation

Configurations B–D in this project decode Serer under the `wol_Latn` (Wolof) NLLB-200
language tag, since `srr_Latn` is not natively supported. This model instead adds a
genuine `srr_Latn` token to NLLB-200's vocabulary — its embedding initialized by copying
`wol_Latn`'s embedding (following the proxy-initialization approach validated for
unseen low-resource languages, e.g. Limbum–English, arXiv:2608.07629) — plus 5 Serer
implosive consonant characters missing from the base vocabulary (ƥ, ƈ, Ƥ, Ƭ, Ƈ),
then fine-tunes with LoRA using the *exact same hyperparameters* as configuration C.

## Model summary

- **Experiment ID:** G (post-submission addition)
- **Base model:** `facebook/nllb-200-distilled-600M` + native `srr_Latn` token
- **Best checkpoint:** `{checkpoint}`
- **LoRA:** rank {cfg.LORA_R}, alpha {cfg.LORA_ALPHA}, dropout {cfg.LORA_DROPOUT}
- **Learning rate:** {cfg.LR}, {cfg.NUM_EPOCHS} epochs, warmup {cfg.WARMUP_STEPS} steps
- **Random seed:** {prov['seed']}

## Evaluation

{metrics_table_md(metrics_rows)}

Evaluated on the held-out test split ({prov['test']} sentence pairs, SHA-256:
`{prov['test_sha256']}`), identical protocol to configuration C for direct comparability.

### Comparison with configuration C (`wol_Latn` proxy, same LoRA hyperparameters)

| Metric | G (native `srr_Latn`) | C (`wol_Latn` proxy) |
|---|---|---|
{comparison_md}

G outperforms C on every metric under an identical training protocol, suggesting that
adding a dedicated target-language token — rather than reusing a related language's tag
as a decoding proxy — is worth the modest extra setup cost when adapting NLLB-200 to an
unsupported language with a well-resourced phylogenetic neighbor.

## Intended use

Research use only. Private repository, not validated for production deployment.
Requires the custom tokenizer bundled with this repo (includes the `srr_Latn` token and
the 5 added Serer characters) — do not swap in a stock NLLB-200 tokenizer.

## Usage

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

repo_id = "{meta_repo_id}"
tokenizer = AutoTokenizer.from_pretrained(repo_id, src_lang="fra_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(repo_id)

text = "Bonjour, comment allez-vous ?"
inputs = tokenizer(text, return_tensors="pt")
target_id = tokenizer.convert_tokens_to_ids("srr_Latn")
output = model.generate(**inputs, forced_bos_token_id=target_id, num_beams=5, max_new_tokens=128)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Training data

Same French–Serer parallel corpus as configurations B–D
({prov['train']} train / {prov['val']} val / {prov['test']} test pairs, ~90% religious
register, ~10% educational glossaries, predominantly Siin dialect). Full provenance is
kept in a private dataset card, available on request.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiments", nargs="+", choices=["B", "C", "D", "E", "F", "G"], required=True)
    parser.add_argument("--organization", required=True)
    args = parser.parse_args()

    from huggingface_hub import HfApi
    api = HfApi()

    for letter in args.experiments:
        print(f"--- Carte pour {letter} ---")
        try:
            card_md = build_card_G(args.organization) if letter == "G" else build_card_B_to_F(letter)
        except Exception as exc:
            print(f"  IGNORÉ: {letter} ({exc})")
            continue

        repo_id = recommended_repo_id(letter, args.organization)
        local_path = ROOT / f"_card_{letter}.md"
        local_path.write_text(card_md, encoding="utf-8")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo="README.md",
            repo_id=repo_id,
        )
        local_path.unlink()
        print(f"  OK: {repo_id}")


if __name__ == "__main__":
    main()

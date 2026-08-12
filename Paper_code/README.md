# French–Serer neural machine translation

Code accompanying *Neural Machine Translation from French to the Low-Resource
Serer Language*. The repository benchmarks six systems and treats the unadapted
NLLB Wolof output as a **proximity probe**, not as a Serer translator.

## Scientific status

The source code now follows the protocol stated in the manuscript. This checkout
does **not** contain trained weights, the independent ~8,500-sentence monolingual
Serer corpus, or completed expert annotations. Consequently, the values in
`paper_results.json` are manuscript-reported values, not results reproduced from
this checkout. Do not remove that distinction when publishing.

| ID | System | Paper protocol |
|---|---|---|
| A | NLLB Wolof-target proximity probe | no training, beam 5 |
| B | NLLB full fine-tuning | 10 epochs, LR 2e-5, batch 32, warmup 500 |
| C | NLLB + LoRA | rank 16, alpha 32, dropout .1, LR 3e-4, 10 epochs, warmup 500 |
| D | C + back-translation | full-FT inverse model, independent SRR mono, 30% synthetic |
| E | OPUS-MT full FT | pretrained French encoder, reinitialized 8k Serer decoder vocabulary, 15 epochs |
| F | Transformer from scratch | 3+3 layers, d=128, FF=256, 4 heads, dropout .3, 30 epochs |

The exact machine-readable settings are in `config.py`. NLLB has no Serer tag;
`wol_Latn` is used as a proxy. Automatic scores, especially BLEU and BERTScore,
must therefore be interpreted together with language identification and expert
assessment.

## Reproduction

Use Python 3.10 or 3.11. From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python prepare_data.py
python baselines/build_tokenizer.py
```

The distributed corpus is treated as the manuscript's already-final 28,892-pair
dataset. Cleaning rules are audited but not reapplied by default, preserving the
reported split sizes. `python prepare_data.py --apply-cleaning` creates a separate
benchmark and must not be used to claim reproduction of Table I.

Training commands:

```bash
python training/train_A_frozen.py
python training/train_B_full.py
python training/train_C_lora.py
python training/train_D_backtranslation.py
python baselines/train_baseline_opusmt.py
python baselines/train_baseline_scratch.py
```

Configuration D additionally requires
`corpus/serere_monolingual.txt`, one independent Serer sentence per line. The
script verifies its minimum size and removes overlap with the parallel train set.
It intentionally fails when that resource is unavailable.

Evaluate and preserve every prediction:

```bash
python evaluate_test.py --configs A B C D
python baselines/evaluate_baselines.py --models opusmt scratch
```

Unified inference for every artifact:

```bash
python inference/translate.py --experiment B --text "Bonjour le monde."
python inference/translate.py --experiment D_REVERSE --text "Une phrase sérère"
python inference/translate.py --experiment F --input phrases.txt --output outputs/F.jsonl
```

The accepted identifiers are `A`, `B`, `C`, `D`, `D_REVERSE`, `E` and `F`.
Configuration A prints an explicit warning because its output is Wolof, not Serer.

Outputs include metrics, SHA-256 dataset fingerprints, environment manifests and
JSONL predictions. Evaluation must run only once after all model choices have
been made on validation data.

## Hugging Face export

Canonical names are defined once in `model_registry.py`. Export is local unless
`--push` is supplied, and remote repositories are private unless `--public` is
explicitly supplied:

```bash
python export_huggingface.py --experiment C
python export_huggingface.py --experiment C --push --organization ORGANIZATION
```

The second command always targets
`ORGANIZATION/french-serer-nllb-lora`. Supported artifacts are B, C, D, the
auxiliary `D_REVERSE`, E and F. A exports only probe metadata/predictions and
never republishes NLLB weights as if they were a Serer model.

Before making anything public, complete `MODEL_CARD.md` and `DATASET_CARD.md`,
choose an appropriate code/data license, verify consent and copyright for the
religious texts, and remove personal or sensitive information.

## Repository layout

- `config.py`: paper-aligned experiment configurations;
- `prepare_data.py`: normalization, filtering, split audit and fingerprints;
- `training/`: A–D training pipelines;
- `baselines/`: E–F tokenizer, models, training and evaluation;
- `evaluate_test.py`: final A–D evaluation and prediction export;
- `annotations/`: expert-evaluation schema;
- `export_huggingface.py`: safe local/HF export;
- `paper_results.json`: reported—not yet reproduced—paper results.

## Reproducibility checklist

- [ ] Pin `MODEL_REVISION` to a Hugging Face commit hash.
- [ ] Archive the exact train/validation/test manifests.
- [ ] Add the independent monolingual Serer corpus or a documented retrieval script.
- [ ] Train every configuration with seed 42 on the documented hardware.
- [ ] Retain checkpoints, logs, predictions and manifests.
- [ ] Complete blinded expert annotation and release the anonymized records.
- [ ] Run multiple seeds and report mean ± standard deviation for stronger claims.
- [ ] Confirm corpus and code licenses before public release.

## Citation and license

Add the final BibTeX entry after acceptance. No license is asserted here because
the corpus rights and intended code license have not yet been documented; GitHub
and Hugging Face publication should remain private until this is resolved.

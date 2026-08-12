---
language:
- fr
- srr
pipeline_tag: translation
tags:
- low-resource
- nllb
- serer
license: other
---

# French–Serer model card template

This card must accompany each exported model. Replace all `TODO` fields before a
public release.

## Model

- Experiment ID: TODO (B/C/D/E)
- Base model and immutable revision: TODO
- Best validation checkpoint: TODO
- Training dataset SHA-256: TODO
- Validation dataset SHA-256: TODO
- Random seed: 42
- Hardware and training duration: TODO

## Intended use

Research on French-to-Serer machine translation in a predominantly religious,
Siin-dialect corpus. The model is not validated for legal, medical, emergency or
fully autonomous publication use.

## Critical limitation

Serer is absent from NLLB-200. NLLB experiments use `wol_Latn` as a decoding
proxy, which can produce Wolof or Wolof-like output and inflate overlap-based
metrics. Human review by a qualified Serer speaker is required.

## Evaluation

Insert reproduced metrics from `results/`, link the exact prediction JSONL, and
report expert assessment. Do not copy values from `paper_results.json` unless the
corresponding artifacts and hashes reproduce them.

## Training data and rights

TODO: document provenance, consent, copyright, dialect, filtering and license.

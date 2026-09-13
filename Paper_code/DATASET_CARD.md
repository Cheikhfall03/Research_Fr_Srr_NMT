---
language:
- fr
- srr
task_categories:
- translation
pretty_name: French–Serer Parallel Corpus
license: other
---

# French–Serer parallel corpus

The manuscript reports 28,892 aligned sentence pairs, approximately 90% biblical
and 10% educational glossary material, primarily in the Siin dialect.

## Processing

`prepare_data.py` normalizes Unicode/whitespace and audits exact duplicates plus
the 1:3–3:1 length rule. The supplied file is treated as the already-filtered
final corpus, so the default preserves the manuscript's 28,892 pairs. The option
`--apply-cleaning` deliberately creates a different, stricter benchmark.

## Limitations

- strong religious-domain concentration;
- primarily one dialect;
- text only, without tone or prosody;
- sentence-random split because document identifiers are unavailable;
- possible near-duplicate biblical passages require an additional audit.

## Provenance and rights (unresolved)

The biblical subset (~90%) and the educational glossaries (~10%) were both
collected from material found online (a digital Bible text and glossary
sources); no documented permission to redistribute has been obtained from
the original rights holder(s) (likely the organization or individual(s)
responsible for the Serer Bible translation). This is an unresolved
copyright question, not a formality: the raw corpus text has been removed
from the public code repository (2026-09-14) pending verification. Only
derived artifacts that do not reproduce the source text -- split sizes and
SHA-256 fingerprints (`data/dataset_manifest.json`), aggregate statistics,
and the fine-tuned model checkpoints -- remain public.

## Release requirements

Before publishing the raw corpus text or declaring an open license for it:
document the exact source of every subset, identify and contact the
copyright holder(s), obtain redistribution permission or use one already
granted for the source, note speaker/translator consent where applicable,
and establish a removal policy and contact information. Until this is
done, the raw corpus text is available only on individual request, and no
open license is declared for it. The accompanying code remains MIT-licensed
regardless (see repository root `LICENSE`).

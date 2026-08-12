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

## Release requirements

Before publication, document the source of every subset, copyright/license,
speaker consent where applicable, removal policy and contact information. Until
then, publish the dataset repository as private or gated and do not declare an
open license.

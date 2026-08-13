# Expérience G — token de langue natif `srr_Latn`

Expérience exploratoire, **hors protocole accepté du papier** (configs A-F).
Teste si donner au sérère un vrai token de langue (au lieu de le faire
décoder sous l'étiquette `wol_Latn` comme les configs A-D) change le
comportement du modèle — notamment la question soulevée par le Reviewer 3 :
la proximité Wolof-Sérère mesurée par le proximity probe (config A) est-elle
un vrai signal de proximité phylogénétique, ou en partie un artefact du fait
qu'on demande littéralement au modèle de produire du wolof ?

Ce dossier est **isolé** du reste du dépôt : checkpoints, résultats et
configuration lui sont propres (`checkpoints/`, `results/` locaux à ce
dossier), pour ne jamais interférer avec le pipeline de révision CNRIA 2026
en cours (`run_full_pipeline.sh`, configs A-F, multi-seed).

## Méthode (voir discussion complète dans l'historique de conversation)

1. **Mesurer d'abord** la couverture du vocabulaire NLLB existant sur le
   corpus sérère (tokens/mot, taux de `<unk>`) — ne construire un nouveau
   tokenizer que si cette mesure est mauvaise (rarement nécessaire d'après
   la littérature: Limbum–English, Tyvan).
2. **Ajouter le token `srr_Latn`** au tokenizer NLLB existant et
   **initialiser son embedding en copiant celui de `wol_Latn`** (proxy
   unique — la moyenne de plusieurs langues apparentées n'apporte pas de
   gain significatif d'après l'étude Limbum, et le wolof est de toute façon
   le seul proxy sérieusement apparenté disponible dans NLLB-200).
3. **Fine-tuner en LoRA** (mêmes hyperparamètres que la config C du papier:
   r=16, alpha=32, dropout=.1, LR=3e-4, 10 epochs, warmup=500) sur les mêmes
   données `data/train.json`/`val.json`/`test.json` — mais avec `srr_Latn`
   comme vraie cible de décodage au lieu du proxy `wol_Latn`.

## Scripts, dans l'ordre

```bash
# 1. Mesurer la couverture du tokenizer NLLB existant sur le corpus sérère
python experiments/srr_token_G/measure_tokenizer_coverage.py

# 2. Ajouter le token srr_Latn et sauvegarder le modèle de base modifié
python experiments/srr_token_G/add_srr_token.py

# 3. Entraîner (LoRA) avec le nouveau token natif
python experiments/srr_token_G/train_G_srr_token.py
```

## Sorties

- `experiments/srr_token_G/coverage_report.json` — résultat de l'étape 1
- `experiments/srr_token_G/base_model_srr/` — modèle NLLB + tokenizer avec
  `srr_Latn` ajouté (étape 2), jamais poussé sur Git (voir `.gitignore`)
- `experiments/srr_token_G/checkpoints/config_G/` — checkpoints LoRA (étape 3)
- `experiments/srr_token_G/results/` — métriques d'évaluation (étape 3)

## Comparaison prévue

Une fois entraîné, comparer au proximity probe (config A, décodage
`wol_Latn` sans fine-tuning) et à la config C (LoRA sur `wol_Latn`) sur le
même test set, pour évaluer si le token natif change le diagnostic de
contamination inter-langues. Cette comparaison n'est *pas* automatisée ici
— à faire manuellement une fois les résultats disponibles.

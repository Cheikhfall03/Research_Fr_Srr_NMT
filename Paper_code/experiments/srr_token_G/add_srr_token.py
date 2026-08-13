"""Ajoute le token de langue srr_Latn à NLLB, initialisé depuis wol_Latn, et
comble les quelques caractères sérères manquants du vocabulaire.

Étape 2 de l'expérience G (voir README.md). Charge le NLLB de base, ajoute
"srr_Latn" comme token spécial, redimensionne les embeddings, puis copie la
ligne d'embedding de "wol_Latn" dans la nouvelle ligne "srr_Latn" (proxy
unique — voir la discussion en README sur pourquoi la moyenne de plusieurs
langues apparentées n'apporte pas de gain net et n'est de toute façon pas
applicable ici, faute d'autres langues Cangin/Atlantic dans NLLB-200).

measure_tokenizer_coverage.py a montré qu'après normalisation de la
ponctuation, le français tombe à 0% de <unk> mais le sérère garde ~0.96%,
entièrement dû à 5 consonnes implosives absentes du vocabulaire NLLB (ƥ, ƈ,
et leurs variantes majuscules) — ƥ seul apparaît >6900 fois dans train.json.
Ce script les ajoute donc aussi, en initialisant chacune depuis la lettre
latine visuellement/phonétiquement la plus proche déjà connue du modèle
(extension minimale et ciblée, pas un nouveau tokenizer complet).

Le modèle et le tokenizer modifiés sont sauvegardés localement; ce script ne
fine-tune rien (voir train_G_srr_token.py pour la suite).

Usage:
    python experiments/srr_token_G/add_srr_token.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, NllbTokenizerFast

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import ProbeConfig  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent / "base_model_srr"
NEW_LANG_TAG = "srr_Latn"
PROXY_LANG_TAG = "wol_Latn"

# Caractères sérères absents du vocabulaire NLLB (mesurés par
# measure_tokenizer_coverage.py) -> lettre latine la plus proche déjà connue,
# utilisée comme point de départ de leur embedding.
MISSING_SERER_CHARS = {
    "ƥ": "p", "Ƥ": "P",
    "ƈ": "c", "Ƈ": "C",
    "ƭ": "t", "Ƭ": "T",
}


def main() -> None:
    cfg = ProbeConfig()
    print(f"Chargement de {cfg.MODEL_NAME} (revision={cfg.MODEL_REVISION})...")
    tokenizer = NllbTokenizerFast.from_pretrained(
        cfg.MODEL_NAME, revision=cfg.MODEL_REVISION, src_lang=cfg.SRC_LANG, tgt_lang=PROXY_LANG_TAG
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.MODEL_NAME, revision=cfg.MODEL_REVISION)

    proxy_id = tokenizer.convert_tokens_to_ids(PROXY_LANG_TAG)
    if proxy_id == tokenizer.unk_token_id:
        raise RuntimeError(f"{PROXY_LANG_TAG} introuvable dans le tokenizer de base — abandon.")

    if NEW_LANG_TAG in tokenizer.additional_special_tokens:
        print(f"{NEW_LANG_TAG} déjà présent dans le tokenizer — rien à ajouter.")
    else:
        n_added = tokenizer.add_special_tokens(
            {"additional_special_tokens": tokenizer.additional_special_tokens + [NEW_LANG_TAG]}
        )
        print(f"{n_added} token ajouté au tokenizer: {NEW_LANG_TAG}")

    model.resize_token_embeddings(len(tokenizer))

    new_id = tokenizer.convert_tokens_to_ids(NEW_LANG_TAG)
    if new_id == tokenizer.unk_token_id:
        raise RuntimeError(f"{NEW_LANG_TAG} n'a pas été correctement enregistré dans le tokenizer.")
    print(f"{PROXY_LANG_TAG} id={proxy_id}  ->  {NEW_LANG_TAG} id={new_id}")

    with torch.no_grad():
        input_embeddings = model.get_input_embeddings()
        input_embeddings.weight[new_id] = input_embeddings.weight[proxy_id].clone()

        output_embeddings = model.get_output_embeddings()
        tied = output_embeddings is not None and output_embeddings.weight is input_embeddings.weight
        if output_embeddings is not None and not tied:
            output_embeddings.weight[new_id] = output_embeddings.weight[proxy_id].clone()

    print(f"\nAjout des {len(MISSING_SERER_CHARS)} caractères sérères absents du vocabulaire...")
    for char, substitute in MISSING_SERER_CHARS.items():
        if char in tokenizer.get_vocab():
            print(f"  {char!r}: déjà présent, ignoré.")
            continue
        n_added = tokenizer.add_tokens([char])
        if n_added == 0:
            print(f"  {char!r}: add_tokens n'a rien ajouté (déjà connu sous une autre forme?), ignoré.")
            continue
        model.resize_token_embeddings(len(tokenizer))
        char_id = tokenizer.convert_tokens_to_ids(char)

        # Référence: moyenne des embeddings de la lettre latine substitute
        # (peut être découpée en plusieurs sous-unités selon le contexte).
        substitute_ids = tokenizer(substitute, add_special_tokens=False)["input_ids"]
        with torch.no_grad():
            input_embeddings = model.get_input_embeddings()
            reference = input_embeddings.weight[substitute_ids].mean(dim=0)
            input_embeddings.weight[char_id] = reference.clone()
            output_embeddings = model.get_output_embeddings()
            if output_embeddings is not None and output_embeddings.weight is not input_embeddings.weight:
                output_embeddings.weight[char_id] = reference.clone()
        print(f"  {char!r} (id={char_id}) initialisé depuis {substitute!r} (ids={substitute_ids})")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nModèle + tokenizer avec {NEW_LANG_TAG} sauvegardés -> {OUTPUT_DIR}")
    print(f"Utilisez MODEL_NAME='{OUTPUT_DIR}' et TGT_LANG='{NEW_LANG_TAG}' pour l'entraînement (train_G_srr_token.py).")


if __name__ == "__main__":
    main()

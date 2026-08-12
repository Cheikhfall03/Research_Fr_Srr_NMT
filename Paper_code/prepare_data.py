"""Prépare et audite les splits officiels français–sérère.

Le script normalise Unicode, retire les doublons exacts, applique le filtre de
rapport de longueur 1:3–3:1 annoncé dans l'article, puis écrit un manifeste avec
les empreintes SHA-256. Le test n'est jamais utilisé pour construire un tokenizer.
"""
from __future__ import annotations

import argparse
import json
import random
import unicodedata
from collections import Counter
from collections import defaultdict
from pathlib import Path

from config import Config
from reproducibility import sha256


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def parse_corpus(path: str) -> list[dict[str, str]]:
    pairs, french = [], None
    with open(path, encoding="utf-8") as stream:
        for raw in stream:
            line = raw.strip()
            if line.startswith("Français :"):
                french = normalize(line.split(":", 1)[1])
            elif line.startswith("Sérère") and french is not None:
                serer = normalize(line.split(":", 1)[1])
                if french and serer:
                    pairs.append({"francais": french, "serere": serer})
                french = None
    return pairs


def clean_pairs(pairs: list[dict[str, str]]) -> tuple[list[dict[str, str]], dict]:
    seen, cleaned = set(), []
    stats = Counter(raw=len(pairs))
    for pair in pairs:
        key = (pair["francais"].casefold(), pair["serere"].casefold())
        if key in seen:
            stats["exact_duplicates"] += 1
            continue
        seen.add(key)
        fr_len = len(pair["francais"].split())
        sr_len = len(pair["serere"].split())
        ratio = fr_len / max(sr_len, 1)
        if not 1 / 3 <= ratio <= 3:
            stats["length_ratio_rejected"] += 1
            continue
        cleaned.append(pair)
    stats["retained"] = len(cleaned)
    return cleaned, dict(stats)


def assert_disjoint(splits: dict[str, list[dict]]) -> None:
    signatures = {
        name: {(x["francais"].casefold(), x["serere"].casefold()) for x in rows}
        for name, rows in splits.items()
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = signatures[left] & signatures[right]
        if overlap:
            raise RuntimeError(f"Fuite de données {left}/{right}: {len(overlap)} paires")


def group_aware_split(pairs: list[dict], cfg: Config) -> dict[str, list[dict]]:
    """Conserve les doublons mais interdit qu'un groupe traverse deux splits."""
    groups = defaultdict(list)
    for pair in pairs:
        groups[(pair["francais"].casefold(), pair["serere"].casefold())].append(pair)
    grouped = list(groups.values())
    random.Random(cfg.SEED).shuffle(grouped)
    targets = {
        "train": int(len(pairs) * cfg.TRAIN_RATIO),
        "val": int(len(pairs) * cfg.VAL_RATIO),
    }
    targets["test"] = len(pairs) - targets["train"] - targets["val"]
    splits = {name: [] for name in targets}
    remaining = grouped
    for name in ("train", "val"):
        chosen, deferred, count = [], [], 0
        for group in remaining:
            if count + len(group) <= targets[name]:
                chosen.extend(group); count += len(group)
            else:
                deferred.append(group)
        if count != targets[name]:
            raise RuntimeError(f"Impossible d'atteindre exactement {targets[name]} lignes pour {name}")
        splits[name] = chosen
        remaining = deferred
    splits["test"] = [pair for group in remaining for pair in group]
    return splits


def split_and_save(pairs: list[dict], cfg: Config, overwrite: bool = False, audit: dict | None = None) -> None:
    output = Path(cfg.DATA_DIR)
    output.mkdir(parents=True, exist_ok=True)
    targets = [output / f"{name}.json" for name in ("train", "val", "test")]
    if not overwrite and any(path.exists() for path in targets):
        raise FileExistsError("Splits existants: utilisez --overwrite pour les remplacer")

    splits = group_aware_split(pairs, cfg)
    assert_disjoint(splits)
    for name, rows in splits.items():
        (output / f"{name}.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    manifest = {
        "seed": cfg.SEED,
        "ratios": {"train": cfg.TRAIN_RATIO, "val": cfg.VAL_RATIO, "test": cfg.TEST_RATIO},
        "counts": {name: len(rows) for name, rows in splits.items()},
        "source_sha256": sha256(cfg.CORPUS_PATH),
        "cleaning_audit": audit or {},
        "splits_sha256": {name: sha256(output / f"{name}.json") for name in splits},
        "warning": "Random sentence split; a document-level split is preferable if document IDs become available.",
    }
    (output / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--apply-cleaning", action="store_true",
        help="Réapplique les filtres et crée un benchmark différent des 28 892 paires de la table I",
    )
    args = parser.parse_args()
    cfg = Config()
    raw = parse_corpus(cfg.CORPUS_PATH)
    normalized = [{"francais": normalize(x["francais"]), "serere": normalize(x["serere"])} for x in raw]
    cleaned, cleaning = clean_pairs(normalized)
    print(f"Audit du corpus distribué: {cleaning}")
    selected = cleaned if args.apply_cleaning else normalized
    split_and_save(selected, cfg, overwrite=args.overwrite, audit=cleaning)


if __name__ == "__main__":
    main()

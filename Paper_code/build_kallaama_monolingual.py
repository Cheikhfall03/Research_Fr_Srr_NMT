"""Build the independent Sereer corpus used by configuration D from Kallaama.

Validated STM transcriptions are preferred. Raw recordings are included only
when no checked version of the same recording exists. The result contains one
utterance per line and excludes exact normalized overlap with the parallel
train/validation/test splits.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import tarfile
import unicodedata
import urllib.request
from pathlib import Path

from config import BackTranslationConfig


KALLAAMA_COMMIT = "abd99f746382364f68c6d6f2acd229623c3e931e"
KALLAAMA_URL = (
    "https://github.com/gauthelo/kallaama-speech-dataset/archive/"
    f"{KALLAAMA_COMMIT}.tar.gz"
)
USER_AGENT = "French-Serer-NMT corpus builder"
LANGUAGE_TAG = re.compile(r"^:[a-z]{2,3}$", re.IGNORECASE)
ATTACHED_LANGUAGE_TAG = re.compile(r":(?:fra|fr|eng|en|ara|ar)$", re.IGNORECASE)


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def clean_transcript(text: str) -> str:
    """Remove STM annotations while retaining the transcribed utterance."""
    cleaned: list[str] = []
    for token in text.split():
        if token.startswith("%") or LANGUAGE_TAG.fullmatch(token):
            continue
        token = ATTACHED_LANGUAGE_TAG.sub("", token)
        if token:
            cleaned.append(token)
    return " ".join(cleaned).strip()


def download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def stm_members(archive: tarfile.TarFile, quality: str) -> dict[str, tarfile.TarInfo]:
    marker = f"/data/transcriptions/{quality}/transcriptions-srr/stm/"
    return {
        Path(member.name).stem: member
        for member in archive.getmembers()
        if member.isfile() and marker in member.name and member.name.endswith(".stm")
    }


def extract_lines(archive_bytes: bytes) -> tuple[list[str], dict]:
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
        checked = stm_members(archive, "checked")
        raw = stm_members(archive, "raw")
        selected = [("checked", item) for item in checked.values()]
        selected.extend(("raw", item) for key, item in raw.items() if key not in checked)

        utterances: list[str] = []
        quality_counts = {"checked": 0, "raw_only": 0}
        for quality, member in sorted(selected, key=lambda item: item[1].name):
            stream = archive.extractfile(member)
            if stream is None:
                continue
            for source_line in stream.read().decode("utf-8-sig").splitlines():
                if not source_line.strip() or source_line.lstrip().startswith(";;"):
                    continue
                fields = source_line.split(maxsplit=6)
                if len(fields) != 7:
                    continue
                sentence = clean_transcript(fields[6])
                if sentence:
                    utterances.append(sentence)
                    quality_counts["checked" if quality == "checked" else "raw_only"] += 1

    return utterances, {
        "checked_recordings": len(checked),
        "raw_recordings": len(raw),
        "selected_raw_only_recordings": len(set(raw) - set(checked)),
        "segments_before_filtering": quality_counts,
    }


def known_parallel_sentences(data_dir: Path) -> set[str]:
    known: set[str] = set()
    for split in ("train", "val", "test"):
        path = data_dir / f"{split}.json"
        if not path.exists():
            raise FileNotFoundError(f"Split requis absent: {path}")
        rows = json.loads(path.read_text(encoding="utf-8"))
        known.update(normalize(row["serere"]) for row in rows)
    return known


def main() -> None:
    cfg = BackTranslationConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Archive Kallaama locale (sinon téléchargement officiel).")
    parser.add_argument("--output", type=Path, default=Path(cfg.MONOLINGUAL_SR_PATH))
    parser.add_argument("--manifest", type=Path, default=Path(cfg.MONOLINGUAL_SR_PATH).with_suffix(".manifest.json"))
    args = parser.parse_args()

    archive_bytes = args.archive.read_bytes() if args.archive else download(KALLAAMA_URL)
    utterances, stats = extract_lines(archive_bytes)
    known = known_parallel_sentences(Path(cfg.DATA_DIR))

    unique: dict[str, str] = {}
    overlap_count = 0
    for sentence in utterances:
        key = normalize(sentence)
        if key in known:
            overlap_count += 1
            continue
        unique.setdefault(key, sentence)

    sentences = list(unique.values())
    if len(sentences) < cfg.EXPECTED_MONOLINGUAL_SIZE:
        raise ValueError(
            f"Corpus Kallaama insuffisant après filtrage: {len(sentences)} "
            f"< {cfg.EXPECTED_MONOLINGUAL_SIZE}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(sentences) + "\n", encoding="utf-8")
    manifest = {
        "source": "Kallaama speech dataset",
        "source_url": "https://github.com/gauthelo/kallaama-speech-dataset",
        "source_commit": KALLAAMA_COMMIT,
        "license": "CC BY 4.0",
        "citation": "Gauthier, Ndiaye and Guissé, RAIL 2024",
        "archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
        **stats,
        "segments_removed_as_parallel_overlap": overlap_count,
        "segments_removed_as_duplicates": len(utterances) - overlap_count - len(sentences),
        "final_sentences": len(sentences),
        "output": str(args.output),
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

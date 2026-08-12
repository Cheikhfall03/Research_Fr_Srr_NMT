"""Utilitaires communs de traçabilité et de reproductibilité."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import lightning as L
import torch


BLEU_PATTERN = re.compile(r"val_bleu[=\-]([0-9]+(?:\.[0-9]+)?)")


def seed_everything(seed: int) -> None:
    L.seed_everything(seed, workers=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def best_checkpoint(directory: str | Path) -> str:
    candidates = [p for p in Path(directory).glob("*.ckpt") if p.name != "last.ckpt"]
    if not candidates:
        raise FileNotFoundError(f"Aucun checkpoint évalué dans {directory}")
    scored = []
    for path in candidates:
        match = BLEU_PATTERN.search(path.name)
        if match:
            scored.append((float(match.group(1)), path))
    if not scored:
        raise ValueError(
            f"Aucun nom de checkpoint ne contient val_bleu dans {directory}; "
            "sélection automatique refusée."
        )
    return str(max(scored, key=lambda item: item[0])[1])


def environment_metadata() -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "command": " ".join(os.sys.argv),
    }


def write_manifest(cfg, output: str | Path, files: list[str | Path] | None = None, **extra) -> None:
    payload = {
        "experiment": cfg.to_dict(),
        "environment": environment_metadata(),
        "files": {
            str(Path(p)): sha256(p) for p in (files or []) if Path(p).is_file()
        },
        **extra,
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

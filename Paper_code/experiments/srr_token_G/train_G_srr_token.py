"""Expérience G — LoRA sur NLLB avec le token de langue natif srr_Latn.

Étape 3 de l'expérience (voir README.md). Reprend exactement les
hyperparamètres LoRA de la config C du papier (r=16, alpha=32, dropout=.1,
LR=3e-4, 10 epochs, warmup=500) et l'infrastructure d'entraînement partagée
(models/model_lora.py, dataset.py), mais charge le modèle de base produit
par add_srr_token.py (avec srr_Latn ajouté) au lieu du NLLB public, et cible
réellement "srr_Latn" au lieu du proxy "wol_Latn" utilisé par les configs
A-D du papier.

Isolé du reste du dépôt: checkpoints et résultats restent dans ce dossier,
jamais dans checkpoints/ ou results/ à la racine (utilisés par le pipeline
de révision CNRIA 2026 en cours).

Prérequis: avoir lancé add_srr_token.py au préalable.

Usage:
    python experiments/srr_token_G/train_G_srr_token.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent
ROOT = EXPERIMENT_DIR.parents[1]
sys.path.insert(0, str(ROOT))

import torch
torch.set_float32_matmul_precision("high")
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger

from config import LoRAExperimentConfig
from reproducibility import seed_everything, write_manifest
from models.model_lora import NLLBFineTuner
from dataset import TranslationDataModule

BASE_MODEL_DIR = EXPERIMENT_DIR / "base_model_srr"


@dataclass
class SrrTokenConfig(LoRAExperimentConfig):
    """Config C avec token de langue natif srr_Latn, isolée dans experiments/srr_token_G/."""
    EXPERIMENT_ID: str = "G"
    MODEL_NAME: str = str(BASE_MODEL_DIR)
    MODEL_REVISION: str = "main"
    TGT_LANG: str = "srr_Latn"
    CHECKPOINTS_DIR: str = str(EXPERIMENT_DIR / "checkpoints")
    RESULTS_DIR: str = str(EXPERIMENT_DIR / "results")
    OUTPUTS_DIR: str = str(EXPERIMENT_DIR / "outputs")
    # BERTScore n'a pas de code "srr" dédié; on garde le proxy wolof comme le
    # fait déjà le reste du dépôt pour cette métrique, à interpréter avec la
    # même prudence.
    BERTSCORE_LANG: str = "wo"

    @property
    def CHECKPOINTS_G(self) -> str:
        return self._seeded_dir("config_G")


def main() -> None:
    if not BASE_MODEL_DIR.exists():
        raise FileNotFoundError(
            f"{BASE_MODEL_DIR} introuvable — lancez d'abord "
            "experiments/srr_token_G/add_srr_token.py."
        )

    cfg = SrrTokenConfig()
    seed_everything(cfg.SEED)
    Path(cfg.CHECKPOINTS_G).mkdir(parents=True, exist_ok=True)
    Path(cfg.RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    n_gpus = torch.cuda.device_count() or 1
    strategy = "ddp" if n_gpus > 1 else "auto"

    model = NLLBFineTuner(cfg)
    datamodule = TranslationDataModule(model.tokenizer, cfg)

    checkpoint_cb = ModelCheckpoint(
        dirpath=cfg.CHECKPOINTS_G,
        filename="french_serer_nllb_lora_srrtoken-epoch={epoch:02d}-val_bleu={val_bleu:.4f}",
        auto_insert_metric_name=False,
        monitor="val_bleu",
        mode="max",
        save_top_k=3,
        save_last=True,
    )
    early_stop_cb = EarlyStopping(monitor="val_bleu", patience=cfg.EARLY_STOPPING_PATIENCE, mode="max")
    logger = CSVLogger(cfg.RESULTS_DIR, name="config_G")

    trainer = L.Trainer(
        max_epochs=cfg.NUM_EPOCHS,
        accelerator="auto",
        devices=n_gpus,
        strategy=strategy,
        precision="16-mixed",
        callbacks=[checkpoint_cb, early_stop_cb],
        logger=logger,
        log_every_n_steps=50,
        val_check_interval=1.0,
        gradient_clip_val=1.0,
        accumulate_grad_batches=cfg.GRAD_ACCUM_STEPS,
    )

    print(f"=== Expérience G : NLLB-200 + LoRA + token natif srr_Latn ({n_gpus} GPU(s)) ===")
    print(f"Modèle de base -> {cfg.MODEL_NAME}")
    print(f"Checkpoints -> {cfg.CHECKPOINTS_G}")
    trainer.fit(model, datamodule=datamodule)
    write_manifest(cfg, Path(cfg.RESULTS_DIR) / "config_G_manifest.json",
                   [Path(cfg.DATA_DIR) / "train.json", Path(cfg.DATA_DIR) / "val.json"],
                   best_checkpoint=checkpoint_cb.best_model_path)
    print(f"Meilleur checkpoint : {checkpoint_cb.best_model_path}")


if __name__ == "__main__":
    main()

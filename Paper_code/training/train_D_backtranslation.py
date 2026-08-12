"""Configuration D conforme au papier: Full FT inverse, puis LoRA FR→SRR.

Ce script exige un corpus monolingue sérère indépendant. Il refuse explicitement
de recycler ``train.json``, car cela ne constituerait pas l'expérience annoncée.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from tqdm import tqdm

from config import BackTranslationConfig, ReverseFullFTConfig
from dataset import TranslationDataModule, TranslationDataset
from models.model_full import NLLBFullFineTuner
from models.model_lora import NLLBFineTuner
from reproducibility import best_checkpoint, seed_everything, write_manifest


# Seed fixe pour le tirage des phrases back-traduites (indépendant du seed
# d'entraînement) afin qu'une étude multi-seed ne varie que l'initialisation/
# l'ordre de mini-batchs du LoRA forward, pas le contenu de l'augmentation.
AUGMENT_SEED = 42

parser = argparse.ArgumentParser(description="Config D (Full FT inverse + LoRA FR->SRR + back-translation)")
parser.add_argument("--seed", type=int, default=None,
                     help="Écrase cfg.SEED pour les runs multi-seed (défaut: 42). "
                          "N'affecte que l'entraînement forward, pas l'augmentation.")
parser.add_argument("--prepare-only", action="store_true",
                     help="Entraîne le modèle inverse, back-traduit et augmente, puis s'arrête "
                          "(pas d'entraînement forward). À lancer UNE SEULE FOIS avant de "
                          "paralléliser plusieurs seeds sur cette config, car ces étapes partagent "
                          "des fichiers (backtranslated.json, train_augmented.json, "
                          "checkpoints/config_D_reverse) qui ne supportent pas l'écriture concurrente.")
args, _unknown = parser.parse_known_args()  # parse_known_args: ce module est importé (pas seulement
                                             # exécuté) par tests/test_urgent_configs.py; parse_args()
                                             # planterait sur des argv étrangers (ex. lancé via pytest).

cfg = BackTranslationConfig()
if args.seed is not None:
    cfg.SEED = args.seed
seed_everything(cfg.SEED)
BT_PATH = Path(cfg.DATA_DIR) / "backtranslated.json"
AUG_PATH = Path(cfg.DATA_DIR) / "train_augmented.json"
REVERSE_DIR = Path(cfg.CHECKPOINTS_DIR) / "config_D_reverse"


class ReverseDataset(TranslationDataset):
    def __getitem__(self, idx):
        item = self.data[idx]
        encoded = self.tokenizer(
            item["serere"], text_target=item["francais"], max_length=self.cfg.MAX_LENGTH,
            truncation=True, padding="max_length", return_tensors="pt",
        )
        labels = encoded["labels"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {"input_ids": encoded["input_ids"].squeeze(),
                "attention_mask": encoded["attention_mask"].squeeze(), "labels": labels}


class ReverseDataModule(TranslationDataModule):
    def setup(self, stage=None):
        self.train_ds = ReverseDataset(str(Path(self.cfg.DATA_DIR) / "train.json"), self.tokenizer, self.cfg)
        self.val_ds = ReverseDataset(str(Path(self.cfg.DATA_DIR) / "val.json"), self.tokenizer, self.cfg)


class AugmentedDataModule(TranslationDataModule):
    def setup(self, stage=None):
        self.train_ds = TranslationDataset(str(AUG_PATH), self.tokenizer, self.cfg)
        self.val_ds = TranslationDataset(str(Path(self.cfg.DATA_DIR) / "val.json"), self.tokenizer, self.cfg)


def trainer_for(cfg_obj, checkpoint_dir, name, patience=None):
    checkpoint = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename=f"{name}-epoch={{epoch:02d}}-val_bleu={{val_bleu:.4f}}",
        auto_insert_metric_name=False, monitor="val_bleu", mode="max", save_top_k=3, save_last=True,
    )
    trainer = L.Trainer(
        max_epochs=cfg_obj.NUM_EPOCHS, accelerator="auto", devices="auto",
        precision="16-mixed", gradient_clip_val=1.0,
        accumulate_grad_batches=cfg_obj.GRAD_ACCUM_STEPS,
        callbacks=[checkpoint, EarlyStopping(
            monitor="val_bleu", patience=patience or cfg_obj.EARLY_STOPPING_PATIENCE, mode="max")],
        logger=CSVLogger(cfg_obj.RESULTS_DIR, name=name), log_every_n_steps=50,
        val_check_interval=1.0,
    )
    return trainer, checkpoint


def train_reverse() -> str:
    REVERSE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        return best_checkpoint(REVERSE_DIR)
    except (FileNotFoundError, ValueError):
        reverse_cfg = ReverseFullFTConfig()
        model = NLLBFullFineTuner(reverse_cfg)
        trainer, callback = trainer_for(reverse_cfg, REVERSE_DIR, "serer_french_nllb_full_ft_backtranslator")
        trainer.fit(model, datamodule=ReverseDataModule(model.tokenizer, reverse_cfg))
        return callback.best_model_path


def normalize_for_overlap(text: str) -> str:
    """Normalisation conservatrice utilisée uniquement pour détecter les fuites."""
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def load_monolingual() -> list[str]:
    path = Path(cfg.MONOLINGUAL_SR_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"Corpus monolingue absent: {path}\n"
            "Ajoutez les ~8 500 phrases sérères indépendantes décrites dans le papier. "
            "Le code refuse d'utiliser les cibles de train.json comme substitut."
        )
    raw_lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    # Déduplication stable après normalisation, en conservant la graphie originale.
    unique = {}
    for line in raw_lines:
        unique.setdefault(normalize_for_overlap(line), line)

    known = set()
    for split in ("train", "val", "test"):
        split_path = Path(cfg.DATA_DIR) / f"{split}.json"
        if not split_path.exists():
            raise FileNotFoundError(f"Split requis absent: {split_path}")
        rows = json.loads(split_path.read_text(encoding="utf-8"))
        known.update(normalize_for_overlap(row["serere"]) for row in rows)

    lines = [original for normalized, original in unique.items() if normalized not in known]
    removed_duplicates = len(raw_lines) - len(unique)
    removed_overlap = len(unique) - len(lines)
    print(
        f"Corpus monolingue: {len(raw_lines):,} brut, {removed_duplicates:,} doublons, "
        f"{removed_overlap:,} chevauchements train/val/test, {len(lines):,} utilisables."
    )
    if len(lines) < cfg.EXPECTED_MONOLINGUAL_SIZE:
        raise ValueError(
            "Corpus monolingue insuffisant après déduplication et exclusion de train/val/test: "
            f"{len(lines)} < {cfg.EXPECTED_MONOLINGUAL_SIZE}"
        )
    return lines


def backtranslate(checkpoint: str) -> None:
    mono = load_monolingual()
    reverse_cfg = ReverseFullFTConfig()
    module = NLLBFullFineTuner.load_from_checkpoint(checkpoint, cfg=reverse_cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    module.eval().to(device)
    french_id = module.tokenizer.convert_tokens_to_ids(reverse_cfg.TGT_LANG)
    pairs = []
    for start in tqdm(range(0, len(mono), cfg.BT_BATCH_SIZE), desc="back-translation"):
        targets = mono[start:start + cfg.BT_BATCH_SIZE]
        inputs = module.tokenizer(targets, return_tensors="pt", padding=True,
                                  truncation=True, max_length=cfg.MAX_LENGTH).to(device)
        with torch.no_grad():
            generated = module.model.generate(**inputs, forced_bos_token_id=french_id,
                                              max_new_tokens=cfg.MAX_LENGTH, num_beams=4)
        french = module.tokenizer.batch_decode(generated, skip_special_tokens=True)
        pairs.extend({"francais": fr.strip(), "serere": sr, "synthetic": True}
                     for fr, sr in zip(french, targets) if fr.strip())
    BT_PATH.write_text(json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8")


def augment() -> None:
    original = json.loads((Path(cfg.DATA_DIR) / "train.json").read_text(encoding="utf-8"))
    synthetic = json.loads(BT_PATH.read_text(encoding="utf-8"))
    count = min(round(len(original) * cfg.BT_RATIO), len(synthetic))
    generator = torch.Generator().manual_seed(AUGMENT_SEED)
    indices = torch.randperm(len(synthetic), generator=generator)[:count].tolist()
    augmented = original + [synthetic[i] for i in indices]
    AUG_PATH.write_text(json.dumps(augmented, ensure_ascii=False, indent=2), encoding="utf-8")


def train_forward() -> str:
    model = NLLBFineTuner(cfg)
    trainer, callback = trainer_for(cfg, cfg.CHECKPOINTS_D, "french_serer_nllb_lora_backtranslation")
    trainer.fit(model, datamodule=AugmentedDataModule(model.tokenizer, cfg))
    return callback.best_model_path


def main() -> None:
    # AUG_PATH déjà présent => la préparation (modèle inverse, back-traduction,
    # augmentation) a déjà tourné, par ce process ou un précédent --prepare-only.
    # On ne la refait pas: ces étapes écrivent des fichiers partagés entre tous
    # les seeds et ne sont pas sûres à exécuter en parallèle (voir --prepare-only).
    if AUG_PATH.exists():
        print(f"Augmentation déjà préparée ({AUG_PATH}) — réutilisation, "
              "pas de nouvel entraînement du modèle inverse ni de back-traduction.")
        checkpoint = str(REVERSE_DIR)
    else:
        checkpoint = train_reverse()
        backtranslate(checkpoint)
        augment()

    if args.prepare_only:
        print("Préparation terminée (--prepare-only): pas d'entraînement forward.")
        return

    best = train_forward()
    manifest_name = "config_D_manifest.json" if cfg.SEED == 42 else f"config_D_manifest_seed{cfg.SEED}.json"
    write_manifest(cfg, Path(cfg.RESULTS_DIR) / manifest_name,
                   [Path(cfg.DATA_DIR) / "train.json", cfg.MONOLINGUAL_SR_PATH, BT_PATH, AUG_PATH],
                   reverse_checkpoint=checkpoint, best_checkpoint=best)


if __name__ == "__main__":
    print(f"=== Config D : Full FT inverse + LoRA + back-translation (seed={cfg.SEED}) ===")
    print(f"Checkpoints -> {cfg.CHECKPOINTS_D}")
    main()

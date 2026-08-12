"""
Config C — NLLB-200 + LoRA (configuration principale).
Checkpoints → checkpoints/config_C/
"""
import argparse
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
torch.set_float32_matmul_precision("high")
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from config import LoRAExperimentConfig
from reproducibility import seed_everything, write_manifest
from models.model_lora import NLLBFineTuner
from dataset import TranslationDataModule

parser = argparse.ArgumentParser(description="Entraînement Config C (NLLB-200 + LoRA)")
parser.add_argument("--seed", type=int, default=None,
                     help="Écrase cfg.SEED pour les runs multi-seed (défaut: 42).")
args, _unknown = parser.parse_known_args()  # parse_known_args: le module est importé (pas seulement
                                             # exécuté) par tests/test_urgent_configs.py; parse_args()
                                             # planterait sur des argv étrangers (ex. lancé via pytest).

cfg = LoRAExperimentConfig()
if args.seed is not None:
    cfg.SEED = args.seed
seed_everything(cfg.SEED)
os.makedirs(cfg.CHECKPOINTS_C, exist_ok=True)
os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

n_gpus = torch.cuda.device_count() or 1
strategy = "ddp" if n_gpus > 1 else "auto"

model      = NLLBFineTuner(cfg)
datamodule = TranslationDataModule(model.tokenizer, cfg)

checkpoint_cb = ModelCheckpoint(
    dirpath=cfg.CHECKPOINTS_C,
    filename="french_serer_nllb_lora-epoch={epoch:02d}-val_bleu={val_bleu:.4f}",
    auto_insert_metric_name=False,
    monitor="val_bleu",
    mode="max",
    save_top_k=3,
    save_last=True,
)
early_stop_cb = EarlyStopping(monitor="val_bleu", patience=cfg.EARLY_STOPPING_PATIENCE, mode="max")
logger = CSVLogger(cfg.RESULTS_DIR, name="config_C")

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

if __name__ == "__main__":
    print(f"=== Config C : NLLB-200 + LoRA (seed={cfg.SEED}, {n_gpus} GPU(s)) ===")
    print(f"Checkpoints -> {cfg.CHECKPOINTS_C}")
    trainer.fit(model, datamodule=datamodule)
    manifest_name = "config_C_manifest.json" if cfg.SEED == 42 else f"config_C_manifest_seed{cfg.SEED}.json"
    write_manifest(cfg, os.path.join(cfg.RESULTS_DIR, manifest_name),
                   [os.path.join(cfg.DATA_DIR, x) for x in ("train.json", "val.json")],
                   best_checkpoint=checkpoint_cb.best_model_path)
    print(f"Meilleur checkpoint : {checkpoint_cb.best_model_path}")

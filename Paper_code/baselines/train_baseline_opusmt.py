"""
Baseline opus-mt-fr — fine-tuning complet sur FR→SRR.
Checkpoints → checkpoints/baseline_opusmt/
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
torch.set_float32_matmul_precision("medium")
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from config import OpusMTConfig
from reproducibility import seed_everything, write_manifest
from baselines.models.model_opusmt import OpusMTFineTuner, BaselineDataModule

cfg = OpusMTConfig()
seed_everything(cfg.SEED)
CKPT_DIR = os.path.join(cfg.CHECKPOINTS_DIR, "baseline_opusmt")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

n_gpus   = torch.cuda.device_count() or 1
strategy = "ddp" if n_gpus > 1 else "auto"

model      = OpusMTFineTuner(cfg)
datamodule = BaselineDataModule(model.source_tokenizer, model.target_tokenizer, cfg)

checkpoint_cb = ModelCheckpoint(
    dirpath=CKPT_DIR,
    filename="french_serer_opusmt_full_ft-epoch={epoch:02d}-val_bleu={val_bleu:.4f}",
    auto_insert_metric_name=False,
    monitor="val_bleu",
    mode="max",
    save_top_k=1,
    save_last=True,
    save_weights_only=True,  # jamais de resume d'entrainement dans ce code, seulement load_from_checkpoint (poids) -> inutile de payer le stockage de l'etat optimizer Adam (~3x la taille du modele)
)
early_stop_cb = EarlyStopping(monitor="val_bleu", patience=cfg.EARLY_STOPPING_PATIENCE, mode="max")
logger = CSVLogger(cfg.RESULTS_DIR, name="baseline_opusmt")

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
)

if __name__ == "__main__":
    print(f"=== Baseline : opus-mt-fr fine-tuné ({n_gpus} GPU(s)) ===")
    print(f"Checkpoints -> {CKPT_DIR}")
    trainer.fit(model, datamodule=datamodule)
    write_manifest(cfg, os.path.join(cfg.RESULTS_DIR, "config_E_manifest.json"),
                   [os.path.join(cfg.DATA_DIR, x) for x in ("train.json", "val.json")],
                   best_checkpoint=checkpoint_cb.best_model_path)
    print(f"Meilleur checkpoint : {checkpoint_cb.best_model_path}")

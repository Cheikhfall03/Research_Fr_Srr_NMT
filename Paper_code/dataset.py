"""
Dataset PyTorch partagé par tous les scripts d'entraînement.
"""
import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
from transformers import NllbTokenizerFast
from config import Config


class TranslationDataset(Dataset):
    def __init__(self, path: str, tokenizer: NllbTokenizerFast, cfg: Config):
        with open(path, encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.cfg = cfg

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        src  = item["francais"]
        tgt  = item["serere"]

        model_inputs = self.tokenizer(
            src,
            text_target=tgt,
            max_length=self.cfg.MAX_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        labels = model_inputs["labels"].squeeze()
        # Remplacer le padding par -100 pour ignorer dans la loss
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids":      model_inputs["input_ids"].squeeze(),
            "attention_mask": model_inputs["attention_mask"].squeeze(),
            "labels":         labels,
        }


class TranslationDataModule(L.LightningDataModule):
    def __init__(self, tokenizer: NllbTokenizerFast, cfg: Config):
        super().__init__()
        self.tokenizer = tokenizer
        self.cfg = cfg

    def setup(self, stage=None):
        self.train_ds = TranslationDataset(
            os.path.join(self.cfg.DATA_DIR, "train.json"), self.tokenizer, self.cfg
        )
        self.val_ds = TranslationDataset(
            os.path.join(self.cfg.DATA_DIR, "val.json"), self.tokenizer, self.cfg
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

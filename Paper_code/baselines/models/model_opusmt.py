"""Configuration E: encodeur OPUS français et vocabulaire cible sérère réinitialisé."""
from __future__ import annotations

import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import evaluate
import lightning as L
import numpy as np
import sentencepiece as spm
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import MarianMTModel, MarianTokenizer

from config import Config


SP_MODEL = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tokenizer/spm.model")
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3


class SererTokenizer:
    def __init__(self, path=SP_MODEL):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Tokenizer cible absent: {path}; lancez baselines/build_tokenizer.py")
        self.sp = spm.SentencePieceProcessor(model_file=path)
        self.vocab_size = self.sp.get_piece_size()
        self.pad_token_id, self.bos_token_id, self.eos_token_id = PAD_ID, BOS_ID, EOS_ID

    def encode(self, text, max_length):
        return (self.sp.encode(text, out_type=int)[:max_length - 1] + [EOS_ID])

    def batch_decode(self, sequences, skip_special_tokens=True):
        rows = sequences.tolist() if torch.is_tensor(sequences) else sequences
        cleaned = [[i for i in row if i not in (PAD_ID, BOS_ID, EOS_ID, -100)] for row in rows]
        return [self.sp.decode(row) for row in cleaned]


class BaselineDataset(Dataset):
    def __init__(self, path, source_tokenizer, target_tokenizer, cfg):
        self.data = json.loads(open(path, encoding="utf-8").read())
        self.source_tokenizer, self.target_tokenizer, self.cfg = source_tokenizer, target_tokenizer, cfg

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        source = self.source_tokenizer(row["francais"], max_length=self.cfg.MAX_LENGTH,
                                       truncation=True, padding="max_length", return_tensors="pt")
        ids = self.target_tokenizer.encode(row["serere"], self.cfg.MAX_LENGTH)
        labels = torch.tensor((ids + [PAD_ID] * self.cfg.MAX_LENGTH)[:self.cfg.MAX_LENGTH])
        labels[labels == PAD_ID] = -100
        return {"input_ids": source["input_ids"].squeeze(0),
                "attention_mask": source["attention_mask"].squeeze(0), "labels": labels}


class BaselineDataModule(L.LightningDataModule):
    def __init__(self, source_tokenizer, target_tokenizer, cfg):
        super().__init__(); self.source_tokenizer = source_tokenizer; self.target_tokenizer = target_tokenizer; self.cfg = cfg

    def setup(self, stage=None):
        self.train_ds = BaselineDataset(os.path.join(self.cfg.DATA_DIR, "train.json"), self.source_tokenizer, self.target_tokenizer, self.cfg)
        self.val_ds = BaselineDataset(os.path.join(self.cfg.DATA_DIR, "val.json"), self.source_tokenizer, self.target_tokenizer, self.cfg)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.cfg.BATCH_SIZE, shuffle=True,
                          num_workers=self.cfg.NUM_WORKERS, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.cfg.BATCH_SIZE, shuffle=False,
                          num_workers=self.cfg.NUM_WORKERS, pin_memory=True)


class OpusMTFineTuner(L.LightningModule):
    def __init__(self, cfg: Config):
        super().__init__(); self.save_hyperparameters(); self.cfg = cfg
        self.source_tokenizer = MarianTokenizer.from_pretrained(
            cfg.MODEL_NAME, revision=cfg.MODEL_REVISION
        )
        self.target_tokenizer = SererTokenizer()
        self.tokenizer = self.source_tokenizer  # compatibilité limitée; décoder avec target_tokenizer
        self.model = MarianMTModel.from_pretrained(
            cfg.MODEL_NAME, revision=cfg.MODEL_REVISION
        )
        hidden = self.model.config.d_model
        vocab = self.target_tokenizer.vocab_size
        # Préserve l'encodeur/source OPUS, réinitialise uniquement la partie cible.
        self.model.model.decoder.embed_tokens = nn.Embedding(vocab, hidden, padding_idx=PAD_ID)
        self.model.lm_head = nn.Linear(hidden, vocab, bias=False)
        nn.init.normal_(self.model.model.decoder.embed_tokens.weight, mean=0.0, std=hidden ** -0.5)
        nn.init.normal_(self.model.lm_head.weight, mean=0.0, std=hidden ** -0.5)
        self.model.final_logits_bias = torch.zeros((1, vocab))
        self.model.config.vocab_size = vocab
        self.model.config.share_encoder_decoder_embeddings = False
        self.model.config.tie_word_embeddings = False
        self.model.config.decoder_start_token_id = BOS_ID
        self.model.config.pad_token_id = PAD_ID
        self.model.config.eos_token_id = EOS_ID
        self.metric_bleu = evaluate.load("sacrebleu")
        self.metric_rouge = evaluate.load("rouge")
        self.metric_bertscore = evaluate.load("bertscore")
        self.validation_step_outputs = []

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        loss = self(**batch).loss; self.log("train_loss", loss, prog_bar=True); return loss

    def validation_step(self, batch, batch_idx):
        output = self(**batch); self.log("val_loss", output.loss, prog_bar=True, sync_dist=True)
        generated = self.model.generate(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                        max_new_tokens=self.cfg.MAX_LENGTH, num_beams=self.cfg.NUM_BEAMS_VAL)
        self.validation_step_outputs.append({
            "preds": self.target_tokenizer.batch_decode(generated),
            "targets": self.target_tokenizer.batch_decode(batch["labels"]),
        })
        return output.loss

    def on_validation_epoch_end(self):
        preds = [p for row in self.validation_step_outputs for p in row["preds"]]
        refs = [p for row in self.validation_step_outputs for p in row["targets"]]
        if not preds: return
        bleu = self.metric_bleu.compute(predictions=preds, references=refs)["score"]
        rouge = self.metric_rouge.compute(predictions=preds, references=refs)["rougeL"]
        bert = np.mean(self.metric_bertscore.compute(predictions=preds, references=refs,
                            model_type=self.cfg.BERTSCORE_MODEL, lang=self.cfg.BERTSCORE_LANG)["f1"])
        # sync_dist=True: en DDP, chaque rank ne voit qu'une moitié du val set ->
        # BLEU local différent par rank sans synchronisation, risque de noms de
        # checkpoint incohérents entre ranks (voir models/model_lora.py).
        self.log_dict({"val_bleu": bleu, "val_rougeL": rouge, "val_bertscore_f1": bert},
                      prog_bar=True, sync_dist=True)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.LR, weight_decay=self.cfg.WEIGHT_DECAY)
        if self.cfg.WARMUP_STEPS <= 0:
            return optimizer
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                                                       total_iters=self.cfg.WARMUP_STEPS)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

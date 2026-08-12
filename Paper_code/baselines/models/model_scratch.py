"""
Baseline Transformer from scratch.
- Tokenizer SentencePiece BPE construit depuis les données FR+SRR (vocab=8000)
- Architecture Transformer standard (PyTorch nn.Transformer)
- Aucun poids pré-entraîné, aucun vocabulaire externe
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json, math
import torch
import torch.nn as nn
import evaluate
import numpy as np
import lightning as L
import sentencepiece as spm
from torch.utils.data import Dataset, DataLoader
from config import Config

TOKENIZER_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tokenizer/spm.model")
PAD_ID, BOS_ID, EOS_ID = 0, 2, 3


# ─── Tokenizer wrapper ────────────────────────────────────────────────────────

class ScratchTokenizer:
    def __init__(self, model_path: str = TOKENIZER_PATH):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        self.pad_token_id = PAD_ID
        self.bos_token_id = BOS_ID
        self.eos_token_id = EOS_ID
        self.vocab_size   = self.sp.get_piece_size()

    def encode(self, text: str, max_length: int = 128) -> list[int]:
        ids = self.sp.encode(text, out_type=int)
        ids = ids[: max_length - 1] + [EOS_ID]
        return ids

    def decode(self, ids: list[int]) -> str:
        ids = [i for i in ids if i not in (PAD_ID, BOS_ID, EOS_ID)]
        return self.sp.decode(ids)

    def pad(self, sequences: list[list[int]], max_length: int) -> tuple:
        padded = [s + [PAD_ID] * (max_length - len(s)) for s in sequences]
        mask   = [[1] * len(s) + [0] * (max_length - len(s)) for s in sequences]
        return padded, mask


# ─── Dataset ──────────────────────────────────────────────────────────────────

class ScratchDataset(Dataset):
    def __init__(self, path: str, tokenizer: ScratchTokenizer, cfg: Config):
        with open(path, encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.cfg = cfg

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        src_ids = self.tokenizer.encode(item["francais"], self.cfg.MAX_LENGTH)
        tgt_ids = [BOS_ID] + self.tokenizer.encode(item["serere"], self.cfg.MAX_LENGTH)

        src_ids = (src_ids + [PAD_ID] * self.cfg.MAX_LENGTH)[: self.cfg.MAX_LENGTH]
        tgt_ids = (tgt_ids + [PAD_ID] * (self.cfg.MAX_LENGTH + 1))[: self.cfg.MAX_LENGTH + 1]

        src_ids = torch.tensor(src_ids, dtype=torch.long)
        tgt_ids = torch.tensor(tgt_ids, dtype=torch.long)

        src_mask = (src_ids != PAD_ID).long()
        labels   = tgt_ids[1:].clone()
        labels[labels == PAD_ID] = -100
        decoder_input = tgt_ids[:-1]

        return {
            "input_ids":      src_ids,
            "attention_mask": src_mask,
            "decoder_input":  decoder_input,
            "labels":         labels,
        }


class ScratchDataModule(L.LightningDataModule):
    def __init__(self, tokenizer: ScratchTokenizer, cfg: Config):
        super().__init__()
        self.tokenizer = tokenizer
        self.cfg = cfg

    def setup(self, stage=None):
        self.train_ds = ScratchDataset(
            os.path.join(self.cfg.DATA_DIR, "train.json"), self.tokenizer, self.cfg
        )
        self.val_ds = ScratchDataset(
            os.path.join(self.cfg.DATA_DIR, "val.json"), self.tokenizer, self.cfg
        )

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.cfg.BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds,   batch_size=self.cfg.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)


# ─── Positional Encoding ──────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1)])


# ─── Seq2Seq Transformer ──────────────────────────────────────────────────────

class Seq2SeqTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model=256, nhead=8,
                 num_enc=3, num_dec=3, ffn=512, dropout=0.1, max_len=256):
        super().__init__()
        self.src_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.tgt_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_enc, num_decoder_layers=num_dec,
            dim_feedforward=ffn, dropout=dropout, batch_first=True,
        )
        self.proj = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_key_padding_mask=None):
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1), device=src.device)
        tgt_pad_mask = (tgt == PAD_ID)
        src_emb = self.pos_enc(self.src_emb(src) * math.sqrt(self.src_emb.embedding_dim))
        tgt_emb = self.pos_enc(self.tgt_emb(tgt) * math.sqrt(self.tgt_emb.embedding_dim))
        out = self.transformer(
            src_emb, tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=(src_key_padding_mask == 0) if src_key_padding_mask is not None else None,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=(src_key_padding_mask == 0) if src_key_padding_mask is not None else None,
        )
        return self.proj(out)

    @torch.no_grad()
    def generate(self, src, attention_mask=None, max_new_tokens=64, num_beams=1):
        if num_beams != 1:
            raise NotImplementedError(
                "Le Transformer scratch utilise le décodage glouton; num_beams doit valoir 1."
            )
        B = src.size(0)
        memory_pad = (attention_mask == 0) if attention_mask is not None else None
        src_emb = self.pos_enc(self.src_emb(src) * math.sqrt(self.src_emb.embedding_dim))
        memory  = self.transformer.encoder(
            src_emb,
            src_key_padding_mask=memory_pad,
        )
        ys = torch.full((B, 1), BOS_ID, dtype=torch.long, device=src.device)
        finished = torch.zeros(B, dtype=torch.bool, device=src.device)
        for _ in range(max_new_tokens):
            tgt_emb = self.pos_enc(self.tgt_emb(ys) * math.sqrt(self.tgt_emb.embedding_dim))
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(ys.size(1), device=src.device)
            out  = self.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask,
                                            memory_key_padding_mask=memory_pad)
            logits   = self.proj(out[:, -1, :])
            next_tok = logits.argmax(-1, keepdim=True)
            # Après le premier EOS d'une phrase, produire uniquement PAD sans
            # modifier les autres phrases encore actives du batch.
            next_tok = torch.where(
                finished.unsqueeze(1), torch.full_like(next_tok, PAD_ID), next_tok
            )
            ys = torch.cat([ys, next_tok], dim=1)
            finished |= next_tok.squeeze(1).eq(EOS_ID)
            if finished.all():
                break
        return ys


# ─── Lightning Module ─────────────────────────────────────────────────────────

class ScratchTransformer(L.LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        if not os.path.exists(TOKENIZER_PATH):
            raise FileNotFoundError(
                f"Tokenizer introuvable : {TOKENIZER_PATH}\n"
                "Lancez d'abord : python baselines/build_tokenizer.py"
            )
        self.tokenizer = ScratchTokenizer(TOKENIZER_PATH)
        self.model = Seq2SeqTransformer(
            self.tokenizer.vocab_size,
            d_model=cfg.D_MODEL,
            nhead=cfg.N_HEADS,
            num_enc=cfg.N_ENCODER_LAYERS,
            num_dec=cfg.N_DECODER_LAYERS,
            ffn=cfg.D_FF,
            dropout=cfg.DROPOUT,
            max_len=cfg.MAX_LENGTH + 2,
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

        self.metric_bleu      = evaluate.load("sacrebleu")
        self.metric_rouge     = evaluate.load("rouge")
        self.metric_bertscore = evaluate.load("bertscore")
        self.validation_step_outputs = []

    def forward(self, src, tgt_in, src_mask=None):
        return self.model(src, tgt_in, src_key_padding_mask=src_mask)

    def training_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["decoder_input"], batch["attention_mask"])
        loss   = self.criterion(logits.reshape(-1, self.tokenizer.vocab_size), batch["labels"].reshape(-1))
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch["input_ids"], batch["decoder_input"], batch["attention_mask"])
        loss   = self.criterion(logits.reshape(-1, self.tokenizer.vocab_size), batch["labels"].reshape(-1))
        self.log("val_loss", loss, prog_bar=True)

        generated = self.model.generate(
            batch["input_ids"], batch["attention_mask"], max_new_tokens=self.cfg.MAX_LENGTH
        )
        preds = [self.tokenizer.decode(g.tolist()) for g in generated]
        refs  = [self.tokenizer.decode(
            torch.where(l != -100, l, torch.tensor(PAD_ID)).tolist()
        ) for l in batch["labels"]]
        self.validation_step_outputs.append({"preds": preds, "targets": refs})
        return loss

    def on_validation_epoch_end(self):
        all_preds   = [p for x in self.validation_step_outputs for p in x["preds"]]
        all_targets = [t for x in self.validation_step_outputs for t in x["targets"]]
        if not all_preds:
            return
        bleu  = self.metric_bleu.compute(predictions=all_preds, references=all_targets)
        rouge = self.metric_rouge.compute(predictions=all_preds, references=all_targets)
        bert  = self.metric_bertscore.compute(
            predictions=all_preds, references=all_targets,
            model_type=self.cfg.BERTSCORE_MODEL, lang=self.cfg.BERTSCORE_LANG,
        )
        self.log("val_bleu",         bleu["score"],        prog_bar=True)
        self.log("val_rouge1",       rouge["rouge1"])
        self.log("val_rougeL",       rouge["rougeL"])
        self.log("val_bertscore_f1", np.mean(bert["f1"]))
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.LR, weight_decay=self.cfg.WEIGHT_DECAY)
        if self.cfg.WARMUP_STEPS <= 0:
            return optimizer
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=self.cfg.WARMUP_STEPS
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

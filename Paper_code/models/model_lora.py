"""Config C & D — NLLB-200 + LoRA (paramètre-efficient fine-tuning)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import evaluate
import numpy as np
import lightning as L
from transformers import AutoModelForSeq2SeqLM, NllbTokenizerFast
from peft import get_peft_model, LoraConfig, TaskType
from config import Config


class NLLBFineTuner(L.LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            cfg.MODEL_NAME, revision=cfg.MODEL_REVISION
        )
        self.tokenizer = NllbTokenizerFast.from_pretrained(
            cfg.MODEL_NAME, revision=cfg.MODEL_REVISION,
            src_lang=cfg.SRC_LANG, tgt_lang=cfg.TGT_LANG
        )

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=cfg.LORA_R,
            lora_alpha=cfg.LORA_ALPHA,
            lora_dropout=cfg.LORA_DROPOUT,
            target_modules=cfg.LORA_TARGETS,
        )
        self.model = get_peft_model(self.model, peft_config)
        trainable, total = self.model.get_nb_trainable_parameters()
        self.trainable_parameters = trainable
        self.total_parameters = total
        print(f"LoRA: {trainable:,}/{total:,} paramètres entraînables ({100*trainable/total:.3f}%)")

        self.metric_bleu      = evaluate.load("sacrebleu")
        self.metric_rouge     = evaluate.load("rouge")
        self.metric_meteor    = evaluate.load("meteor")
        self.metric_bertscore = evaluate.load("bertscore")

        self.validation_step_outputs = []

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        loss = self(batch["input_ids"], batch["attention_mask"], batch["labels"]).loss
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch["input_ids"], batch["attention_mask"], batch["labels"])
        self.log("val_loss", outputs.loss, prog_bar=True)

        tgt_lang_id = self.tokenizer.convert_tokens_to_ids(self.cfg.TGT_LANG)
        generated = self.model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            forced_bos_token_id=tgt_lang_id,
            max_new_tokens=self.cfg.MAX_LENGTH,
            num_beams=self.cfg.NUM_BEAMS_VAL,
        )
        preds  = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        labels = torch.where(
            batch["labels"] != -100, batch["labels"], self.tokenizer.pad_token_id
        )
        refs = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        self.validation_step_outputs.append({"preds": preds, "targets": refs})
        return outputs.loss

    def on_validation_epoch_end(self):
        all_preds   = [p for x in self.validation_step_outputs for p in x["preds"]]
        all_targets = [t for x in self.validation_step_outputs for t in x["targets"]]
        if not all_preds:
            return

        bleu   = self.metric_bleu.compute(predictions=all_preds, references=all_targets)
        rouge  = self.metric_rouge.compute(predictions=all_preds, references=all_targets)
        meteor = self.metric_meteor.compute(predictions=all_preds, references=all_targets)
        bert   = self.metric_bertscore.compute(
            predictions=all_preds,
            references=all_targets,
            model_type=self.cfg.BERTSCORE_MODEL,
            lang=self.cfg.BERTSCORE_LANG,
        )
        self.log("val_bleu",        bleu["score"],        prog_bar=True)
        self.log("val_rouge1",       rouge["rouge1"])
        self.log("val_rougeL",       rouge["rougeL"])
        self.log("val_meteor",       meteor["meteor"])
        self.log("val_bertscore_f1", np.mean(bert["f1"]))
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.cfg.LR, weight_decay=self.cfg.WEIGHT_DECAY
        )
        if self.cfg.WARMUP_STEPS <= 0:
            return optimizer
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=self.cfg.WARMUP_STEPS
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

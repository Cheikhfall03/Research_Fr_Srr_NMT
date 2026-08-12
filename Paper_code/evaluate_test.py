"""Évaluation reproductible A–D sur le test, avec conservation des prédictions."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import evaluate
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, NllbTokenizerFast

from config import FullFTConfig, LoRAExperimentConfig, ProbeConfig, BackTranslationConfig
from reproducibility import best_checkpoint, sha256, write_manifest


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def metrics(predictions, references, cfg):
    bleu = evaluate.load("sacrebleu").compute(predictions=predictions, references=references)["score"]
    chrf = evaluate.load("chrf").compute(predictions=predictions, references=references)["score"]
    rouge = evaluate.load("rouge").compute(predictions=predictions, references=references)
    bert = evaluate.load("bertscore").compute(predictions=predictions, references=references,
                                               model_type=cfg.BERTSCORE_MODEL, lang=cfg.BERTSCORE_LANG)
    return {"bleu": round(bleu, 4), "chrf": round(chrf, 4),
            "rouge1": round(rouge["rouge1"], 4), "rougeL": round(rouge["rougeL"], 4),
            "bertscore_f1": round(float(np.mean(bert["f1"])), 4)}


def generate(model, tokenizer, rows, cfg):
    predictions, losses = [], []
    target_id = tokenizer.convert_tokens_to_ids(cfg.TGT_LANG)
    for start in tqdm(range(0, len(rows), cfg.BATCH_SIZE), desc=cfg.EXPERIMENT_ID):
        batch = rows[start:start + cfg.BATCH_SIZE]
        source = [x["francais"] for x in batch]
        target = [x["serere"] for x in batch]
        encoded = tokenizer(source, text_target=target, return_tensors="pt", padding=True,
                            truncation=True, max_length=cfg.MAX_LENGTH).to(DEVICE)
        labels = encoded["labels"].clone(); labels[labels == tokenizer.pad_token_id] = -100
        with torch.no_grad():
            losses.append(model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"], labels=labels).loss.item())
            output = model.generate(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"],
                                    forced_bos_token_id=target_id, max_new_tokens=cfg.MAX_LENGTH,
                                    num_beams=cfg.NUM_BEAMS_TEST)
        predictions.extend(tokenizer.batch_decode(output, skip_special_tokens=True))
    return predictions, round(float(np.mean(losses)), 4)


def load_experiment(name, seed=None):
    if name == "A":
        cfg = ProbeConfig()
        tokenizer = NllbTokenizerFast.from_pretrained(cfg.MODEL_NAME, revision=cfg.MODEL_REVISION,
                                                       src_lang=cfg.SRC_LANG, tgt_lang=cfg.TGT_LANG)
        return cfg, AutoModelForSeq2SeqLM.from_pretrained(cfg.MODEL_NAME, revision=cfg.MODEL_REVISION), tokenizer, None
    if name == "B":
        from models.model_full import NLLBFullFineTuner
        cfg = FullFTConfig()
        if seed is not None: cfg.SEED = seed
        checkpoint = best_checkpoint(cfg.CHECKPOINTS_B)
        module = NLLBFullFineTuner.load_from_checkpoint(checkpoint, cfg=cfg)
    elif name == "C":
        from models.model_lora import NLLBFineTuner
        cfg = LoRAExperimentConfig()
        if seed is not None: cfg.SEED = seed
        checkpoint = best_checkpoint(cfg.CHECKPOINTS_C)
        module = NLLBFineTuner.load_from_checkpoint(checkpoint, cfg=cfg)
    else:
        from models.model_lora import NLLBFineTuner
        cfg = BackTranslationConfig()
        if seed is not None: cfg.SEED = seed
        checkpoint = best_checkpoint(cfg.CHECKPOINTS_D)
        module = NLLBFineTuner.load_from_checkpoint(checkpoint, cfg=cfg)
    return cfg, module.model, module.tokenizer, checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", choices=list("ABCD"), default=list("ABCD"))
    parser.add_argument("--seed", type=int, default=None,
                         help="Charge le checkpoint entraîné avec ce seed (config_X_seedN) "
                              "et suffixe les fichiers de sortie en conséquence. Ignoré pour A.")
    args = parser.parse_args()
    suffix = "" if args.seed is None else f"_seed{args.seed}"
    base = ProbeConfig(); test_path = Path(base.DATA_DIR) / "test.json"
    rows = json.loads(test_path.read_text(encoding="utf-8"))
    all_scores = {}
    for name in args.configs:
        cfg, model, tokenizer, checkpoint = load_experiment(name, seed=args.seed)
        model.eval().to(DEVICE)
        predictions, loss = generate(model, tokenizer, rows, cfg)
        refs = [x["serere"] for x in rows]
        score = {**metrics(predictions, refs, cfg), "loss": loss,
                 "checkpoint": checkpoint, "test_sha256": sha256(test_path), "seed": cfg.SEED}
        all_scores[name] = score
        prediction_rows = [{"id": i, "francais": row["francais"], "serere_reference": row["serere"],
                            "prediction": pred} for i, (row, pred) in enumerate(zip(rows, predictions))]
        Path(cfg.OUTPUTS_DIR).mkdir(parents=True, exist_ok=True)
        (Path(cfg.OUTPUTS_DIR) / f"predictions_{name}{suffix}.jsonl").write_text(
            "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in prediction_rows), encoding="utf-8")
        write_manifest(cfg, Path(cfg.RESULTS_DIR) / f"evaluation_{name}{suffix}_manifest.json", [test_path], metrics=score)
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    results = Path(base.RESULTS_DIR); results.mkdir(parents=True, exist_ok=True)
    (results / f"evaluation_test{suffix}.json").write_text(json.dumps(all_scores, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = ["config", "bleu", "chrf", "rouge1", "rougeL", "bertscore_f1", "loss", "checkpoint", "test_sha256", "seed"]
    with open(results / f"evaluation_test{suffix}.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for name, score in all_scores.items(): writer.writerow({"config": name, **score})


if __name__ == "__main__":
    main()

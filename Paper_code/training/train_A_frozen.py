"""
Config A — NLLB-200 Frozen (sans adaptation).
Évalue directement le modèle de base sur val et test.
Sortie : results/config_A_metrics.json
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch
import numpy as np
import evaluate
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, NllbTokenizerFast
from config import ProbeConfig

cfg = ProbeConfig()
os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device : {device}")

print("Chargement du modèle NLLB-200 (frozen)...")
tokenizer = NllbTokenizerFast.from_pretrained(
    cfg.MODEL_NAME, revision=cfg.MODEL_REVISION,
    src_lang=cfg.SRC_LANG, tgt_lang=cfg.TGT_LANG
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    cfg.MODEL_NAME, revision=cfg.MODEL_REVISION
).to(device)
model.eval()

tgt_lang_id = tokenizer.convert_tokens_to_ids(cfg.TGT_LANG)


def translate_batch(sources: list[str]) -> list[str]:
    inputs = tokenizer(
        sources,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=cfg.MAX_LENGTH,
    ).to(device)
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            forced_bos_token_id=tgt_lang_id,
            max_new_tokens=cfg.MAX_LENGTH,
            num_beams=5,
        )
    return tokenizer.batch_decode(generated, skip_special_tokens=True)


def evaluate_split(split: str) -> dict:
    path = os.path.join(cfg.DATA_DIR, f"{split}.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    metric_bleu      = evaluate.load("sacrebleu")
    metric_rouge     = evaluate.load("rouge")
    metric_bertscore = evaluate.load("bertscore")

    all_preds, all_refs = [], []
    for i in tqdm(range(0, len(data), cfg.BATCH_SIZE), desc=f"  [{split}]"):
        batch   = data[i : i + cfg.BATCH_SIZE]
        sources = [x["francais"] for x in batch]
        refs    = [x["serere"]   for x in batch]
        preds   = translate_batch(sources)
        all_preds.extend(preds)
        all_refs.extend(refs)

    bleu  = metric_bleu.compute(predictions=all_preds, references=all_refs)
    rouge = metric_rouge.compute(predictions=all_preds, references=all_refs)
    bert  = metric_bertscore.compute(
        predictions=all_preds,
        references=all_refs,
        model_type=cfg.BERTSCORE_MODEL,
        lang=cfg.BERTSCORE_LANG,
    )
    return {
        "bleu":         round(bleu["score"],              4),
        "rouge1":       round(rouge["rouge1"],            4),
        "rougeL":       round(rouge["rougeL"],            4),
        "bertscore_f1": round(float(np.mean(bert["f1"])), 4),
        "n_examples":   len(data),
    }


if __name__ == "__main__":
    print("\nEvaluation Config A — sonde de proximité ciblant explicitement le wolof")
    results = {}
    for split in ("val", "test"):
        print(f"\nSplit : {split}")
        results[split] = evaluate_split(split)
        for k, v in results[split].items():
            print(f"  {k}: {v}")

    out_path = os.path.join(cfg.RESULTS_DIR, "config_A_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"config": "A_frozen", **results}, f, ensure_ascii=False, indent=2)
    print(f"\nResultats sauvegardes : {out_path}")

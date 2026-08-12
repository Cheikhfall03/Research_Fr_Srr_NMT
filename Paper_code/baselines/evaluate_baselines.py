"""
Évaluation finale des baselines sur le split TEST.

Pour chaque baseline (scratch, opusmt) :
  - Charge le meilleur checkpoint
  - Génère les traductions sur test.json
  - Calcule BLEU, ROUGE-1, ROUGE-L, BERTScore F1, Loss
  - Sauvegarde dans results/evaluation_baselines.json et .csv

Usage :
    python baselines/evaluate_baselines.py [--models scratch opusmt]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse, json, csv
import torch.nn as nn
import numpy as np
import torch
import evaluate
from tqdm import tqdm
from config import Config, OpusMTConfig, ScratchConfig
from reproducibility import best_checkpoint

cfg    = Config()
BOS_ID = 2
device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
print(f"Device : {device}\n")

CKPT_DIRS = {
    "scratch": os.path.join(cfg.CHECKPOINTS_DIR, "baseline_scratch"),
    "opusmt":  os.path.join(cfg.CHECKPOINTS_DIR, "baseline_opusmt"),
}


def load_test_data():
    with open(os.path.join(cfg.DATA_DIR, "test.json"), encoding="utf-8") as f:
        return json.load(f)


def compute_metrics(preds, refs):
    m_bleu  = evaluate.load("sacrebleu")
    m_rouge = evaluate.load("rouge")
    m_bert  = evaluate.load("bertscore")
    bleu  = m_bleu.compute(predictions=preds, references=refs)
    rouge = m_rouge.compute(predictions=preds, references=refs)
    bert  = m_bert.compute(predictions=preds, references=refs,
                           model_type=cfg.BERTSCORE_MODEL, lang=cfg.BERTSCORE_LANG)
    return {
        "bleu":         round(bleu["score"],              4),
        "rouge1":       round(rouge["rouge1"],            4),
        "rougeL":       round(rouge["rougeL"],            4),
        "bertscore_f1": round(float(np.mean(bert["f1"])), 4),
    }


def translate_all(model, tokenizer, data, desc, forced_bos=None):
    preds = []
    for i in tqdm(range(0, len(data), cfg.BATCH_SIZE), desc=f"  {desc}"):
        batch   = data[i : i + cfg.BATCH_SIZE]
        sources = [x["francais"] for x in batch]
        inputs  = tokenizer(sources, return_tensors="pt", padding=True,
                            truncation=True, max_length=cfg.MAX_LENGTH).to(device)
        gen_kwargs = dict(max_new_tokens=cfg.MAX_LENGTH, num_beams=5)
        if forced_bos is not None:
            gen_kwargs["forced_bos_token_id"] = forced_bos
        with torch.no_grad():
            generated = model.generate(**inputs, **gen_kwargs)
        preds.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    return preds


def compute_loss(model, tokenizer, data):
    total_loss, n = 0.0, 0
    for i in range(0, len(data), cfg.BATCH_SIZE):
        batch   = data[i : i + cfg.BATCH_SIZE]
        sources = [x["francais"] for x in batch]
        targets = [x["serere"]   for x in batch]
        inputs  = tokenizer(sources, text_target=targets, return_tensors="pt",
                            padding=True, truncation=True, max_length=cfg.MAX_LENGTH).to(device)
        labels = inputs["labels"].clone()
        labels[labels == tokenizer.pad_token_id] = -100
        with torch.no_grad():
            loss = model(input_ids=inputs["input_ids"],
                         attention_mask=inputs["attention_mask"],
                         labels=labels).loss
        total_loss += loss.item()
        n += 1
    return round(total_loss / n, 4)


def eval_scratch(data):
    from baselines.models.model_scratch import ScratchTransformer, PAD_ID
    print("Baseline scratch — Transformer from scratch")
    local_cfg = ScratchConfig()
    ckpt = best_checkpoint(CKPT_DIRS["scratch"])
    print(f"  checkpoint : {os.path.basename(ckpt)}")
    pl = ScratchTransformer.load_from_checkpoint(ckpt, cfg=local_cfg)
    pl.eval().to(device)
    tokenizer = pl.tokenizer
    refs  = [x["serere"] for x in data]
    preds = []
    for i in tqdm(range(0, len(data), cfg.BATCH_SIZE), desc="  scratch"):
        batch   = data[i : i + cfg.BATCH_SIZE]
        sources = [x["francais"] for x in batch]
        enc     = [tokenizer.encode(s, cfg.MAX_LENGTH) for s in sources]
        padded, mask = tokenizer.pad(enc, cfg.MAX_LENGTH)
        src  = torch.tensor(padded, dtype=torch.long).to(device)
        amsk = torch.tensor(mask,   dtype=torch.long).to(device)
        with torch.no_grad():
            gen = pl.model.generate(src, amsk, max_new_tokens=cfg.MAX_LENGTH)
        preds.extend([tokenizer.decode(g.tolist()) for g in gen])
    metrics = compute_metrics(preds, refs)
    # loss
    total, n = 0.0, 0
    for i in range(0, len(data), cfg.BATCH_SIZE):
        batch   = data[i : i + cfg.BATCH_SIZE]
        enc_src = [tokenizer.encode(x["francais"], cfg.MAX_LENGTH) for x in batch]
        enc_tgt = [[BOS_ID] + tokenizer.encode(x["serere"], cfg.MAX_LENGTH) for x in batch]
        padded_src, mask_src = tokenizer.pad(enc_src, cfg.MAX_LENGTH)
        padded_tgt, _        = tokenizer.pad(enc_tgt, cfg.MAX_LENGTH + 1)
        src = torch.tensor(padded_src, dtype=torch.long).to(device)
        amk = torch.tensor(mask_src,   dtype=torch.long).to(device)
        tgt = torch.tensor(padded_tgt, dtype=torch.long).to(device)
        tgt_in = tgt[:, :-1]; labels = tgt[:, 1:].clone()
        labels[labels == PAD_ID] = -100
        with torch.no_grad():
            logits = pl.model(src, tgt_in, amk)
            loss   = nn.CrossEntropyLoss(ignore_index=-100)(
                logits.reshape(-1, tokenizer.vocab_size), labels.reshape(-1)
            )
        total += loss.item(); n += 1
    metrics["loss"] = round(total / n, 4)
    del pl; torch.cuda.empty_cache()
    return metrics


def eval_opusmt(data):
    from baselines.models.model_opusmt import OpusMTFineTuner
    print("Baseline opusmt — opus-mt-fr fine-tuné")
    local_cfg = OpusMTConfig()
    ckpt = best_checkpoint(CKPT_DIRS["opusmt"])
    print(f"  checkpoint : {os.path.basename(ckpt)}")
    pl = OpusMTFineTuner.load_from_checkpoint(ckpt, cfg=local_cfg)
    pl.eval().to(device)
    refs, preds, losses = [x["serere"] for x in data], [], []
    for i in tqdm(range(0, len(data), local_cfg.BATCH_SIZE), desc="  opusmt"):
        batch = data[i:i + local_cfg.BATCH_SIZE]
        enc = pl.source_tokenizer([x["francais"] for x in batch], return_tensors="pt", padding=True,
                                  truncation=True, max_length=local_cfg.MAX_LENGTH).to(device)
        target_ids = [pl.target_tokenizer.encode(x["serere"], local_cfg.MAX_LENGTH) for x in batch]
        labels = torch.full((len(batch), local_cfg.MAX_LENGTH), -100, dtype=torch.long, device=device)
        for row, ids in enumerate(target_ids): labels[row, :len(ids)] = torch.tensor(ids, device=device)
        with torch.no_grad():
            losses.append(pl.model(**enc, labels=labels).loss.item())
            generated = pl.model.generate(**enc, max_new_tokens=local_cfg.MAX_LENGTH,
                                          num_beams=local_cfg.NUM_BEAMS_TEST)
        preds.extend(pl.target_tokenizer.batch_decode(generated))
    metrics = compute_metrics(preds, refs)
    metrics["loss"] = round(float(np.mean(losses)), 4)
    del pl; torch.cuda.empty_cache()
    return metrics


EVAL_MAP = {"scratch": eval_scratch, "opusmt": eval_opusmt}

LABELS = {
    "scratch": "Transformer scratch",
    "opusmt":  "opus-mt-fr fine-tuné",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["scratch", "opusmt"],
                        choices=["scratch", "opusmt"])
    args = parser.parse_args()

    data = load_test_data()
    print(f"Test set : {len(data):,d} exemples\n")

    all_results = {}
    for name in args.models:
        try:
            metrics = EVAL_MAP[name](data)
            all_results[name] = metrics
            print(f"  → {metrics}\n")
        except FileNotFoundError as e:
            print(f"  IGNORÉ : {e}\n")

    json_path = os.path.join(cfg.RESULTS_DIR, "evaluation_baselines.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"Résultats JSON → {json_path}")

    csv_path = os.path.join(cfg.RESULTS_DIR, "evaluation_baselines.csv")
    headers = ["modele", "bleu", "rouge1", "rougeL", "bertscore_f1", "loss"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for name, m in all_results.items():
            writer.writerow({"modele": LABELS[name], **m})
    print(f"Résultats CSV  → {csv_path}")

    print("\n" + "=" * 72)
    print(f"{'Modèle':<24} {'BLEU':>7} {'ROUGE-1':>8} {'ROUGE-L':>8} {'BERTScore':>10} {'Perte':>7}")
    print("-" * 72)
    for name, m in all_results.items():
        print(f"{LABELS[name]:<24} {m['bleu']:>7.4f} {m['rouge1']:>8.4f} "
              f"{m['rougeL']:>8.4f} {m['bertscore_f1']:>10.4f} {m.get('loss',0):>7.4f}")
    print("=" * 72)


if __name__ == "__main__":
    main()

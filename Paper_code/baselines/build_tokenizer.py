"""
Construit un tokenizer SentencePiece (BPE) depuis les données d'entraînement FR+SRR.
Sauvegarde dans baselines/tokenizer/spm.model et spm.vocab

Usage :
    python baselines/build_tokenizer.py
Prérequis :
    data/train.json  (créé par prepare_data.py)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import sentencepiece as spm
from config import Config

cfg = Config()
TOKENIZER_DIR = os.path.join(os.path.dirname(__file__), "tokenizer")
os.makedirs(TOKENIZER_DIR, exist_ok=True)

corpus_path = os.path.join(TOKENIZER_DIR, "corpus_combined.txt")

print("Construction du corpus FR+SRR...")
with open(os.path.join(cfg.DATA_DIR, "train.json"), encoding="utf-8") as f:
    data = json.load(f)

with open(corpus_path, "w", encoding="utf-8") as out:
    for item in data:
        out.write(item["francais"].strip() + "\n")
        out.write(item["serere"].strip()   + "\n")

print(f"  {len(data)*2:,d} phrases écrites dans {corpus_path}")

print("Entraînement SentencePiece (BPE, vocab=8000)...")
spm.SentencePieceTrainer.train(
    input=corpus_path,
    model_prefix=os.path.join(TOKENIZER_DIR, "spm"),
    vocab_size=8000,
    character_coverage=1.0,
    model_type="bpe",
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
    pad_piece="<pad>",
    unk_piece="<unk>",
    bos_piece="<s>",
    eos_piece="</s>",
)

print(f"Tokenizer sauvegardé : {os.path.join(TOKENIZER_DIR, 'spm.model')}")
print("Done.")

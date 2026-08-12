"""
Inference — Config A : NLLB-200 frozen (sans adaptation).
Traduit 20 exemples du split test → outputs/resultats_config_A.docx
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch
from transformers import AutoModelForSeq2SeqLM, NllbTokenizerFast
from docx import Document
from config import ProbeConfig

cfg = ProbeConfig()
os.makedirs(cfg.OUTPUTS_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device : {device}")
print("Chargement NLLB-200 (aucun fine-tuning)...")

tokenizer = NllbTokenizerFast.from_pretrained(
    cfg.MODEL_NAME, revision=cfg.MODEL_REVISION,
    src_lang=cfg.SRC_LANG, tgt_lang=cfg.TGT_LANG
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    cfg.MODEL_NAME, revision=cfg.MODEL_REVISION
).to(device)
model.eval()

tgt_lang_id = tokenizer.convert_tokens_to_ids(cfg.TGT_LANG)


def translate(text: str) -> str:
    if not text:
        return ""
    inputs = tokenizer(
        text, return_tensors="pt", padding=True,
        truncation=True, max_length=cfg.MAX_LENGTH,
    ).to(device)
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            forced_bos_token_id=tgt_lang_id,
            max_new_tokens=cfg.MAX_LENGTH,
            num_beams=5,
        )
    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]


with open(os.path.join(cfg.DATA_DIR, "test.json"), encoding="utf-8") as f:
    samples = json.load(f)[:20]

doc = Document()
doc.add_heading("Résultats — Config A : sonde NLLB ciblant le wolof", 0)
doc.add_paragraph(
    "Sonde de proximité, pas traducteur sérère : NLLB reçoit explicitement le tag wol_Latn."
)
table = doc.add_table(rows=1, cols=3)
table.style = "Table Grid"
hdr = table.rows[0].cells
hdr[0].text = "Phrase source (francais)"
hdr[1].text = "Sortie prédite (wolof attendu)"
hdr[2].text = "Reference (serere)"

print(f"\nTraduction de {len(samples)} exemples...\n")
for item in samples:
    src, ref = item["francais"], item["serere"]
    pred = translate(src)
    print(f"FR  : {src}\nSRR : {pred}\nREF : {ref}\n" + "-" * 50)
    row = table.add_row().cells
    row[0].text, row[1].text, row[2].text = src, pred, ref

output_path = os.path.join(cfg.OUTPUTS_DIR, "resultats_config_A.docx")
doc.save(output_path)
print(f"\nDocument Word cree : {output_path}")

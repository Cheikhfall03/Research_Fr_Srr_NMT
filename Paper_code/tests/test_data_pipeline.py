"""Tests légers sans téléchargement de modèles."""
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prepare_data import clean_pairs, normalize, assert_disjoint
from reproducibility import BLEU_PATTERN
from model_registry import model_metadata, recommended_repo_id


assert normalize("e\u0301   test") == "é test"
cleaned, stats = clean_pairs([
    {"francais": "bonjour monde", "serere": "foo bar"},
    {"francais": "bonjour monde", "serere": "foo bar"},
    {"francais": "un deux trois quatre", "serere": "x"},
])
assert len(cleaned) == 1
assert stats["exact_duplicates"] == 1
assert stats["length_ratio_rejected"] == 1
assert BLEU_PATTERN.search("model-epoch=03-val_bleu=24.0800.ckpt").group(1) == "24.0800"
assert_disjoint({"train": cleaned, "val": [], "test": []})
assert model_metadata("D_REVERSE")["kind"] == "auxiliary_backtranslation_model"
assert recommended_repo_id("C", "serer-nmt") == "serer-nmt/french-serer-nllb-lora"
assert model_metadata("A")["publish_model"] is False
print("data pipeline tests: OK")

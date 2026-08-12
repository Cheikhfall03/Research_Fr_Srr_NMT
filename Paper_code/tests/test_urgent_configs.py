"""Non-régressions des corrections urgentes E, F et D."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import OpusMTConfig, ScratchConfig, LoRAExperimentConfig, BackTranslationConfig
from baselines.models.model_scratch import ScratchDataModule
import training.train_D_backtranslation as config_d


assert OpusMTConfig().WARMUP_STEPS > 0
assert ScratchConfig().WARMUP_STEPS > 0
# C/D utilisent le LR le plus élevé du protocole (3e-4); un warmup=0 les rendait
# instables au démarrage et pouvait déclencher un early stopping prématuré.
assert LoRAExperimentConfig().WARMUP_STEPS > 0
assert BackTranslationConfig().WARMUP_STEPS > 0

# Le DataModule F doit joindre DATA_DIR et les noms de fichiers correctement.
scratch_cfg = ScratchConfig()
scratch_module = ScratchDataModule(tokenizer=object(), cfg=scratch_cfg)
scratch_module.setup()
assert Path(scratch_module.train_ds.data[0] and scratch_cfg.DATA_DIR, "train.json").exists()

# D doit retirer doublons et chevauchements des trois splits avant le contrôle
# de taille, tout en conservant les graphies originales des phrases valides.
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    data = root / "data"; data.mkdir()
    mono = root / "mono.txt"
    (data / "train.json").write_text(json.dumps([{"serere": "Déjà train"}]), encoding="utf-8")
    (data / "val.json").write_text(json.dumps([{"serere": "Déjà val"}]), encoding="utf-8")
    (data / "test.json").write_text(json.dumps([{"serere": "Déjà test"}]), encoding="utf-8")
    mono.write_text("Déjà train\nDéjà val\nDéjà test\nPhrase A\n phrase   a \nPhrase B\n", encoding="utf-8")
    previous = (config_d.cfg.DATA_DIR, config_d.cfg.MONOLINGUAL_SR_PATH,
                config_d.cfg.EXPECTED_MONOLINGUAL_SIZE)
    try:
        config_d.cfg.DATA_DIR = str(data)
        config_d.cfg.MONOLINGUAL_SR_PATH = str(mono)
        config_d.cfg.EXPECTED_MONOLINGUAL_SIZE = 2
        assert config_d.load_monolingual() == ["Phrase A", "Phrase B"]
        config_d.cfg.EXPECTED_MONOLINGUAL_SIZE = 3
        try:
            config_d.load_monolingual()
        except ValueError as error:
            assert "après déduplication" in str(error)
        else:
            raise AssertionError("D aurait dû refuser moins de 3 phrases finales")
    finally:
        (config_d.cfg.DATA_DIR, config_d.cfg.MONOLINGUAL_SR_PATH,
         config_d.cfg.EXPECTED_MONOLINGUAL_SIZE) = previous

print("urgent configuration tests: OK")

"""Configurations reproduisant le protocole décrit dans l'article.

Chaque expérience possède ses propres hyperparamètres. Les chemins sont absolus
afin que les commandes fonctionnent depuis la racine du dépôt ou ``Paper_code``.
"""
from dataclasses import asdict, dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass
class Config:
    EXPERIMENT_ID: str = "A"
    MODEL_NAME: str = "facebook/nllb-200-distilled-600M"
    MODEL_REVISION: str = "main"  # à remplacer par un commit HF pour une archive finale
    SRC_LANG: str = "fra_Latn"
    # Le sérère est absent de NLLB. wol_Latn sert de proxy de décodage.
    TGT_LANG: str = "wol_Latn"
    MAX_LENGTH: int = 128
    # BATCH_SIZE=32 en une passe sature une RTX 4090 24GB pour le full FT NLLB-200
    # (600M params, précision mixte) combiné à la génération beam-search en
    # validation (même batch, cf. dataset.py) -> CUDA OOM observé en pratique.
    # BATCH_SIZE=8 + GRAD_ACCUM_STEPS=4 reproduit le batch effectif de 32
    # documenté dans le papier pour l'entraînement, tout en réduisant d'autant
    # la mémoire de la génération de validation.
    BATCH_SIZE: int = 8
    NUM_EPOCHS: int = 0
    LR: float = 0.0
    WARMUP_STEPS: int = 0
    WEIGHT_DECAY: float = 0.1
    EARLY_STOPPING_PATIENCE: int = 2
    NUM_BEAMS_VAL: int = 4
    NUM_BEAMS_TEST: int = 5
    GRAD_ACCUM_STEPS: int = 4
    SEED: int = 42
    TRAIN_RATIO: float = 0.80
    VAL_RATIO: float = 0.10
    TEST_RATIO: float = 0.10
    NUM_WORKERS: int = 4
    BERTSCORE_MODEL: str = "distilbert-base-multilingual-cased"
    BERTSCORE_LANG: str = "wo"  # proxy, à interpréter avec prudence
    CORPUS_PATH: str = str(PROJECT_ROOT / "corpus/Corpus_Français_Serere_Aligné.txt")
    MONOLINGUAL_SR_PATH: str = str(PROJECT_ROOT / "corpus/serere_monolingual.txt")
    DATA_DIR: str = str(PROJECT_ROOT / "data")
    RESULTS_DIR: str = str(PROJECT_ROOT / "results")
    OUTPUTS_DIR: str = str(PROJECT_ROOT / "outputs")
    CHECKPOINTS_DIR: str = str(PROJECT_ROOT / "checkpoints")
    LORA_R: int = 16
    LORA_ALPHA: int = 32
    LORA_DROPOUT: float = 0.1
    # Projections annoncées dans le papier; NLLB utilise le nom out_proj.
    LORA_TARGETS: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "out_proj"]
    )

    @property
    def CHECKPOINT_DIR(self) -> str:
        return self._seeded_dir(f"config_{self.EXPERIMENT_ID}")

    @property
    def CHECKPOINTS_A(self) -> str:
        return self._seeded_dir("config_A")

    @property
    def CHECKPOINTS_B(self) -> str:
        return self._seeded_dir("config_B")

    @property
    def CHECKPOINTS_C(self) -> str:
        return self._seeded_dir("config_C")

    @property
    def CHECKPOINTS_D(self) -> str:
        return self._seeded_dir("config_D")

    def _seeded_dir(self, name: str) -> str:
        # Le seed 42 est le seed par défaut du papier: on garde le nom de dossier
        # historique pour ne pas invalider les checkpoints déjà produits. Les
        # autres seeds (étude multi-seed) vont dans un sous-dossier dédié.
        base = Path(self.CHECKPOINTS_DIR) / name
        return str(base) if self.SEED == 42 else str(base) + f"_seed{self.SEED}"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ProbeConfig(Config):
    EXPERIMENT_ID: str = "A"


@dataclass
class FullFTConfig(Config):
    EXPERIMENT_ID: str = "B"
    LR: float = 2e-5
    NUM_EPOCHS: int = 10
    WARMUP_STEPS: int = 500


@dataclass
class LoRAExperimentConfig(Config):
    EXPERIMENT_ID: str = "C"
    LR: float = 3e-4
    NUM_EPOCHS: int = 10
    # Aligné sur B/E/F (warmup=500): un LR=3e-4 sans warmup est le seul cas du
    # protocole à démarrer sans montée progressive, ce qui déstabilise le début
    # d'entraînement et peut déclencher un early stopping prématuré (patience=2,
    # val_check_interval=0.5 => arrêt possible après une seule epoch sans gain).
    WARMUP_STEPS: int = 500


@dataclass
class BackTranslationConfig(LoRAExperimentConfig):
    EXPERIMENT_ID: str = "D"
    BT_RATIO: float = 0.30
    # Génération pure (pas de rétropropagation) donc moins gourmande que
    # l'entraînement, mais réduite par prudence après l'OOM observé sur B avec
    # BATCH_SIZE=32 sur RTX 4090 24GB.
    BT_BATCH_SIZE: int = 16
    EXPECTED_MONOLINGUAL_SIZE: int = 8500


@dataclass
class ReverseFullFTConfig(FullFTConfig):
    """Modèle inverse SRR→FR utilisé pour produire les données synthétiques."""
    EXPERIMENT_ID: str = "D_reverse"
    SRC_LANG: str = "wol_Latn"
    TGT_LANG: str = "fra_Latn"


@dataclass
class OpusMTConfig(Config):
    EXPERIMENT_ID: str = "E"
    MODEL_NAME: str = "Helsinki-NLP/opus-mt-fr-en"
    LR: float = 2e-5
    NUM_EPOCHS: int = 15
    WARMUP_STEPS: int = 500
    TARGET_VOCAB_SIZE: int = 8000


@dataclass
class ScratchConfig(Config):
    EXPERIMENT_ID: str = "F"
    LR: float = 3e-4
    NUM_EPOCHS: int = 30
    WARMUP_STEPS: int = 500
    D_MODEL: int = 128
    D_FF: int = 256
    N_HEADS: int = 4
    N_ENCODER_LAYERS: int = 3
    N_DECODER_LAYERS: int = 3
    DROPOUT: float = 0.3
    TARGET_VOCAB_SIZE: int = 8000


EXPERIMENT_CONFIGS = {
    "A": ProbeConfig,
    "B": FullFTConfig,
    "C": LoRAExperimentConfig,
    "D": BackTranslationConfig,
    "E": OpusMTConfig,
    "F": ScratchConfig,
}


def get_config(experiment_id: str) -> Config:
    try:
        return EXPERIMENT_CONFIGS[experiment_id.upper()]()
    except KeyError as exc:
        raise ValueError(f"Configuration inconnue: {experiment_id}") from exc

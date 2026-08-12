#!/bin/bash
# Pipeline complet pour révision CNRIA 2026 — à lancer sur un pod RunPod.
#
# Enchaîne: setup, préparation des données, tokenizer, entraînement de tous
# les systèmes (B, C, D multi-seed, E, F), toutes les évaluations (test set,
# baselines, calibration UDHR hors-domaine), le test de significativité par
# bootstrap, les statistiques de longueur/format d'entrée, et la génération
# du package de correction pour un second annotateur. Chaque étape écrit son
# propre log sous results/pipeline_logs/ et le pipeline s'arrête à la
# première erreur (le log correspondant indique où).
#
# Usage:
#   bash run_full_pipeline.sh                     # tout, du setup à l'agrégation
#   bash run_full_pipeline.sh --skip-setup         # dépendances déjà installées
#   bash run_full_pipeline.sh --seeds "43 44 45"   # seeds ADDITIONNELS pour le multi-seed
#                                                   # (42 est déjà entraîné/évalué aux étapes
#                                                   # 06-07-11 et réutilisé, jamais dupliqué)
#   bash run_full_pipeline.sh --gpus "0 1"         # GPU explicites pour le multi-seed
#   bash run_full_pipeline.sh --clean-checkpoints  # supprime les checkpoints seed-42
#                                                   # B/C/D/E/F avant réentraînement
#                                                   # (nécessaire si le pod contient déjà
#                                                   # des checkpoints C/D entraînés avec
#                                                   # l'ancien warmup=0, corrigé depuis)
#
# Prérequis: corpus/Corpus_Français_Serere_Aligné.txt et
# corpus/serere_monolingual.txt déjà présents dans le repo cloné sur le pod
# (le corpus n'est pas re-téléchargé par ce script).

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

SKIP_SETUP=false
SEEDS="43 44"  # seeds additionnels seulement; 42 vient des étapes 06-07-11
GPUS=""
CLEAN_CHECKPOINTS=false
LOG_DIR="results/pipeline_logs"
mkdir -p "$LOG_DIR"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-setup) SKIP_SETUP=true; shift ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --clean-checkpoints) CLEAN_CHECKPOINTS=true; shift ;;
    *) echo "Argument inconnu: $1"; exit 1 ;;
  esac
done

STEP=0
run_step() {
  STEP=$((STEP + 1))
  local name="$1"; shift
  local log_file="${LOG_DIR}/${STEP}_${name}.log"
  echo ""
  echo "=== [Étape ${STEP}] ${name} ==="
  echo "commande: $*"
  echo "log: ${log_file}"
  local start_ts=$(date +%s)
  if "$@" > "${log_file}" 2>&1; then
    local elapsed=$(( $(date +%s) - start_ts ))
    echo "OK (${elapsed}s)"
  else
    echo "ECHEC — voir ${log_file}"
    tail -n 40 "${log_file}"
    exit 1
  fi
}

echo "########################################################"
echo "# Pipeline complet — Neural MT French->Serer (CNRIA 2026)"
echo "# GPUs visibles:"
python -c "import torch; print(f'  {torch.cuda.device_count()} GPU(s)'); [print(f'  - {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]" || true
echo "########################################################"

# --- -1. Vérification amont des prérequis --------------------------------
# Le corpus parallèle et le probe UDHR ne peuvent pas être régénérés par ce
# script (pas de source de téléchargement automatique) — ils doivent déjà être
# dans le repo cloné sur le pod. Le corpus monolingue sérère, lui, peut être
# reconstruit automatiquement depuis Kallaama (voir étape 01b ci-dessous).
for f in "corpus/Corpus_Français_Serere_Aligné.txt" "data/udhr_fr_srr_probe_test.json"; do
  if [[ ! -f "$f" ]]; then
    echo "ERREUR: fichier requis absent: $f"
    echo "Ce script ne télécharge pas le corpus — vérifiez que le repo cloné sur le pod le contient."
    exit 1
  fi
done

# --- 0. Environnement ---------------------------------------------------
if [[ "$SKIP_SETUP" == false ]]; then
  run_step "00_setup_runpod" bash setup_runpod.sh
else
  echo "[Setup ignoré: --skip-setup]"
fi

if [[ "$CLEAN_CHECKPOINTS" == true ]]; then
  echo ""
  echo "=== Nettoyage des checkpoints seed-42 (config.py corrigé: warmup C/D 0 -> 500) ==="
  for dir in checkpoints/config_B checkpoints/config_C checkpoints/config_D \
             checkpoints/config_D_reverse checkpoints/baseline_opusmt checkpoints/baseline_scratch; do
    if [[ -d "$dir" ]]; then
      echo "  suppression: $dir"
      rm -rf "$dir"
    fi
  done
  echo "  résultats/prédictions seed-42 précédents également supprimés."
  rm -f results/evaluation_test.json results/evaluation_test.csv
  rm -f outputs/predictions_A.jsonl outputs/predictions_B.jsonl outputs/predictions_C.jsonl outputs/predictions_D.jsonl \
        outputs/predictions_E.jsonl outputs/predictions_F.jsonl
fi

# --- 1. Données -----------------------------------------------------------
run_step "01_prepare_data" python prepare_data.py --overwrite
if [[ ! -f "corpus/serere_monolingual.txt" ]]; then
  run_step "01b_build_kallaama_monolingual" python build_kallaama_monolingual.py
else
  echo "[corpus/serere_monolingual.txt déjà présent — construction Kallaama ignorée]"
fi
run_step "02_check_pipeline" python -m compileall -q .
run_step "03_test_data_pipeline" python tests/test_data_pipeline.py
run_step "03b_test_urgent_configs" python tests/test_urgent_configs.py
run_step "04_tokenizer" python baselines/build_tokenizer.py

# --- 2. Entraînement seed 42 (référence, comme dans le papier) ------------
run_step "05_train_B_full" python training/train_B_full.py
run_step "06_train_C_lora" python training/train_C_lora.py
run_step "07_train_D_backtranslation" python training/train_D_backtranslation.py
run_step "08_train_E_opusmt" python baselines/train_baseline_opusmt.py
run_step "09_train_F_scratch" python baselines/train_baseline_scratch.py

# --- 3. Évaluation seed 42 (référence) -------------------------------------
# Doit précéder le multi-seed: run_multiseed.py relit results/evaluation_test.json
# (seed 42, sans suffixe) pour l'inclure dans l'agrégat sans le réentraîner.
run_step "10_evaluate_test_ABCD" python evaluate_test.py --configs A B C D
run_step "11_evaluate_baselines_EF" python baselines/evaluate_baselines.py --models scratch opusmt
run_step "12_calibration_udhr" python evaluate_udhr_calibration.py --configs A B C D

# --- 4. Multi-seed C & D (réponse à R2/R3 sur la robustesse statistique) --
if [[ -n "$GPUS" ]]; then
  run_step "13_multiseed_C_D" python training/run_multiseed.py --configs C D --seeds ${SEEDS} --gpus ${GPUS}
else
  run_step "13_multiseed_C_D" python training/run_multiseed.py --configs C D --seeds ${SEEDS}
fi

# --- 5. Significativité statistique (réponse à R3 sur le gap C/D) ---------
run_step "14_bootstrap_C_vs_D" python bootstrap_significance.py C D
run_step "15_bootstrap_B_vs_C" python bootstrap_significance.py B C
run_step "16_bootstrap_E_vs_C" python bootstrap_significance.py E C
run_step "17_bootstrap_E_vs_D" python bootstrap_significance.py E D

# --- 6. Longueurs de phrases / format d'entrée (réponse à R1) -------------
run_step "18_corpus_statistics" python corpus_statistics.py

# --- 7. Package pour un second correcteur (réponse à R1/R2/R3 sur l'annotateur unique) ---
run_step "19_correctors_package" python build_correctors_package.py --configs B C D E --n-samples 30 --include-udhr

# --- 8. Archive finale ------------------------------------------------------
ARCHIVE="revision_cnria2026_$(date +%Y%m%d_%H%M).tar.gz"
tar -czf "${ARCHIVE}" results/ annotations/round2_correctors/ outputs/ 2>/dev/null || \
  tar -czf "${ARCHIVE}" results/ annotations/round2_correctors/

# --- 9. Dépôt sur Google Drive (best-effort, non bloquant) -----------------
# Nécessite rclone configuré sur CETTE machine avec un remote "gdrive:"
# (copiez ~/.config/rclone/rclone.conf depuis votre machine locale vers le
# pod si ce n'est pas déjà fait). N'échoue jamais le pipeline: les résultats
# restent de toute façon dans l'archive locale ci-dessus.
STEP=$((STEP + 1))
echo ""
echo "=== [Étape ${STEP}] 20_deploy_drive ==="
bash deploy_to_drive.sh 2>&1 | tee "${LOG_DIR}/${STEP}_20_deploy_drive.log" || \
  echo "Dépôt Drive échoué ou ignoré — voir ${LOG_DIR}/${STEP}_20_deploy_drive.log (non bloquant)."

# --- 10. Publication des meilleurs modèles sur Hugging Face (best-effort) --
# Nécessite une session HF active sur CETTE machine (huggingface-cli login,
# ou variable HF_TOKEN). Dépôts privés par défaut (HF_ORG=Fallovski).
# N'échoue jamais le pipeline: ignore proprement les configs sans checkpoint.
STEP=$((STEP + 1))
echo ""
echo "=== [Étape ${STEP}] 21_deploy_huggingface ==="
HF_ORG="${HF_ORG:-Fallovski}" bash deploy_to_huggingface.sh 2>&1 | tee "${LOG_DIR}/${STEP}_21_deploy_huggingface.log" || \
  echo "Dépôt Hugging Face échoué ou ignoré — voir ${LOG_DIR}/${STEP}_21_deploy_huggingface.log (non bloquant)."

echo ""
echo "########################################################"
echo "# Pipeline terminé avec succès."
echo "# Archive des résultats: ${ARCHIVE}"
echo "# Logs détaillés: ${LOG_DIR}/"
echo "########################################################"

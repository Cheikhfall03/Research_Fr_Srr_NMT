#!/bin/bash
# Publie les meilleurs modèles (B, C, D, E, F) sur le Hugging Face Hub.
#
# Réutilise export_huggingface.py (déjà dans le repo): export local puis push
# vers ORGANIZATION/<slug-canonique> (voir model_registry.py), dépôts privés
# par défaut. Ignore proprement (sans faire échouer le pipeline) toute config
# dont le checkpoint n'existe pas encore ou si aucun token HF n'est configuré.
#
# Prérequis: authentification Hugging Face sur CETTE machine (le pod), via
# soit `huggingface-cli login`, soit la variable d'environnement HF_TOKEN.
# Ce script ne transporte aucun identifiant.
#
# Usage:
#   HF_ORG=Fallovski bash deploy_to_huggingface.sh
#   HF_ORG=Fallovski HF_PUBLIC=true bash deploy_to_huggingface.sh   # dépôts publics

set -uo pipefail  # pas de -e: une config manquante ne doit pas interrompre les autres
cd "$(dirname "${BASH_SOURCE[0]}")"

HF_ORG="${HF_ORG:-Fallovski}"
PUBLIC_FLAG=""
[[ "${HF_PUBLIC:-false}" == "true" ]] && PUBLIC_FLAG="--public"

if ! python3 -c "import huggingface_hub" 2>/dev/null; then
  echo "[deploy_to_huggingface] huggingface_hub non installé — dépôt HF ignoré."
  exit 0
fi
if ! python3 -c "
from huggingface_hub import HfApi
try:
    HfApi().whoami()
except Exception as e:
    raise SystemExit(1)
" 2>/dev/null; then
  echo "[deploy_to_huggingface] Pas de session Hugging Face active (huggingface-cli login ou HF_TOKEN manquant) — dépôt HF ignoré."
  exit 0
fi

echo "=== [deploy_to_huggingface] Organisation cible: ${HF_ORG} (${PUBLIC_FLAG:-privé}) ==="

STATUS=0
for experiment in B C D E F G; do
  echo ""
  echo "--- Export + push ${experiment} ---"
  if python3 export_huggingface.py --experiment "${experiment}" --push --organization "${HF_ORG}" ${PUBLIC_FLAG}; then
    echo "  OK: ${experiment}"
  else
    echo "  IGNORÉ: ${experiment} (checkpoint absent ou erreur d'export — voir sortie ci-dessus)"
    STATUS=1
  fi
done

echo ""
echo "[deploy_to_huggingface] Terminé (voir ci-dessus pour le détail par config)."
exit 0  # jamais bloquant pour le pipeline global, même si certaines configs ont été ignorées
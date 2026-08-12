#!/bin/bash
# Dépose les meilleurs checkpoints et les résultats sur Google Drive via rclone.
#
# Ne fait rien (sortie propre, code 0) si rclone n'est pas installé ou si le
# remote n'est pas configuré sur cette machine — pensé pour être appelé en fin
# de run_full_pipeline.sh sans jamais faire échouer le pipeline à cause de ça.
#
# Prérequis: rclone configuré sur CETTE machine (le pod) avec un remote nommé
# "gdrive:" pointant vers le même compte Google Drive. Ce script ne configure
# pas rclone lui-même et ne transporte aucun identifiant — copiez votre
# ~/.config/rclone/rclone.conf local vers le pod vous-même, ex.:
#   scp ~/.config/rclone/rclone.conf <user>@<pod-host>:~/.config/rclone/rclone.conf
#
# Usage:
#   bash deploy_to_drive.sh
#   DRIVE_REMOTE="gdrive:AutreDossier" bash deploy_to_drive.sh

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

DRIVE_REMOTE="${DRIVE_REMOTE:-gdrive:CNRIA2026_Revision}"

if ! command -v rclone &> /dev/null; then
  echo "[deploy_to_drive] rclone introuvable sur cette machine — dépôt Drive ignoré."
  exit 0
fi
if ! rclone listremotes 2>/dev/null | grep -q "^${DRIVE_REMOTE%%:*}:$"; then
  echo "[deploy_to_drive] Remote '${DRIVE_REMOTE%%:*}:' non configuré (rclone config) — dépôt Drive ignoré."
  exit 0
fi

echo "=== [deploy_to_drive] Repérage des meilleurs checkpoints ==="
python3 - <<'PYEOF'
import glob
import sys
from pathlib import Path

sys.path.insert(0, ".")
from reproducibility import best_checkpoint

dirs = {
    "B": "checkpoints/config_B",
    "C": "checkpoints/config_C",
    "D": "checkpoints/config_D",
    "E": "checkpoints/baseline_opusmt",
    "F": "checkpoints/baseline_scratch",
}
for path in glob.glob("checkpoints/config_C_seed*") + glob.glob("checkpoints/config_D_seed*"):
    dirs[Path(path).name] = path

found = {}
for label, directory in dirs.items():
    if not Path(directory).exists():
        continue
    try:
        found[label] = best_checkpoint(directory)
    except (FileNotFoundError, ValueError) as exc:
        print(f"  {label}: ignoré ({exc})")

Path("results").mkdir(parents=True, exist_ok=True)
Path("results/best_checkpoints_manifest.txt").write_text(
    "\n".join(f"{label}\t{path}" for label, path in found.items()), encoding="utf-8"
)
for label, path in found.items():
    print(f"  {label}: {Path(path).name}")
PYEOF

if [[ -s results/best_checkpoints_manifest.txt ]]; then
  echo "=== [deploy_to_drive] Upload des meilleurs checkpoints -> ${DRIVE_REMOTE}/checkpoints/ ==="
  while IFS=$'\t' read -r label path; do
    [[ -z "${label:-}" ]] && continue
    echo "  upload ${label}: $(basename "$path")"
    rclone copy "$path" "${DRIVE_REMOTE}/checkpoints/${label}/"
  done < results/best_checkpoints_manifest.txt
else
  echo "[deploy_to_drive] Aucun checkpoint trouvé — rien à uploader (entraînement pas encore terminé ?)."
fi

echo "=== [deploy_to_drive] Upload des résultats -> ${DRIVE_REMOTE}/results/ ==="
rclone copy results/ "${DRIVE_REMOTE}/results/" --exclude "*/version_*/**"

if [[ -d annotations/round2_correctors ]]; then
  echo "=== [deploy_to_drive] Upload du package correcteur -> ${DRIVE_REMOTE}/annotations/round2_correctors/ ==="
  rclone copy annotations/round2_correctors "${DRIVE_REMOTE}/annotations/round2_correctors/"
fi

echo "[deploy_to_drive] Terminé."

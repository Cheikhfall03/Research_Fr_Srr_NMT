"""Orchestre les runs multi-seed pour C et D en parallèle sur plusieurs GPU.

Répond à la remarque des reviewers sur le seed unique (42): pour les configs
contestées (C = LoRA, D = LoRA+back-translation), entraîne et évalue sur
plusieurs seeds, puis rapporte moyenne et écart-type par métrique. Chaque
seed a son propre dossier de checkpoints (config_X_seedN, voir config.py)
donc les runs existants en seed 42 ne sont jamais écrasés.

Parallélisation: chaque run (config, seed) est un job indépendant assigné à
un seul GPU via CUDA_VISIBLE_DEVICES (data-parallélisme "embarrassingly
parallel" entre seeds, distinct du DDP intra-run déjà utilisé par
train_C_lora.py/train_D_backtranslation.py quand ils tournent seuls sur
plusieurs GPU). Avec N GPU disponibles, jusqu'à N jobs tournent simultanément;
le reste est mis en file d'attente.

Usage:
    python training/run_multiseed.py --configs C D --seeds 42 43 44
    python training/run_multiseed.py --configs C D --seeds 42 43 44 --gpus 0 1
    python training/run_multiseed.py --configs C D --seeds 42 43 44 --skip-training
"""
from __future__ import annotations

import argparse
import csv
import json
import queue
import subprocess
import sys
import threading
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = {"C": ROOT / "training" / "train_C_lora.py", "D": ROOT / "training" / "train_D_backtranslation.py"}
METRICS = ["bleu", "chrf", "rouge1", "rougeL", "bertscore_f1"]


def detect_gpu_count() -> int:
    try:
        import torch
        return max(torch.cuda.device_count(), 1)
    except Exception:
        return 1


def run(cmd: list[str], env: dict | None = None, log_path: Path | None = None) -> None:
    print(f"+ {' '.join(cmd)}" + (f"  (CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES')})" if env else ""))
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as log_file:
            subprocess.run(cmd, check=True, cwd=ROOT, env=env, stdout=log_file, stderr=subprocess.STDOUT)
    else:
        subprocess.run(cmd, check=True, cwd=ROOT, env=env)


def run_jobs_on_gpu_pool(jobs: list[tuple[str, int]], gpu_ids: list[int], logs_dir: Path) -> None:
    """Exécute (config, seed) jobs, au plus len(gpu_ids) en parallèle, un GPU par job."""
    import os

    job_queue: "queue.Queue[tuple[str, int]]" = queue.Queue()
    for job in jobs:
        job_queue.put(job)
    errors: list[str] = []
    lock = threading.Lock()

    def worker(gpu_id: int) -> None:
        while True:
            try:
                config, seed = job_queue.get_nowait()
            except queue.Empty:
                return
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            log_path = logs_dir / f"train_{config}_seed{seed}_gpu{gpu_id}.log"
            try:
                run([sys.executable, str(TRAIN_SCRIPT[config]), "--seed", str(seed)], env=env, log_path=log_path)
                print(f"[GPU {gpu_id}] OK: config {config}, seed {seed} (log: {log_path})")
            except subprocess.CalledProcessError as exc:
                with lock:
                    errors.append(f"config {config} seed {seed} sur GPU {gpu_id}: {exc}")
                print(f"[GPU {gpu_id}] ECHEC: config {config}, seed {seed} — voir {log_path}")
            finally:
                job_queue.task_done()

    threads = [threading.Thread(target=worker, args=(gpu_id,), daemon=True) for gpu_id in gpu_ids]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    if errors:
        raise RuntimeError("Certains runs multi-seed ont échoué:\n" + "\n".join(errors))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", nargs="+", choices=["C", "D"], default=["C", "D"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--gpus", nargs="+", type=int, default=None,
                         help="IDs de GPU à utiliser (défaut: tous les GPU détectés).")
    parser.add_argument("--skip-training", action="store_true",
                         help="Réutilise des checkpoints déjà entraînés et ne fait que l'évaluation/agrégation.")
    args = parser.parse_args()

    gpu_ids = args.gpus if args.gpus else list(range(detect_gpu_count()))
    logs_dir = ROOT / "results" / "multiseed_logs"

    if not args.skip_training:
        if "D" in args.configs:
            # Étape partagée (modèle inverse + back-traduction + augmentation) à faire une
            # seule fois, séquentiellement: les seeds D en parallèle écriraient tous dans
            # les mêmes fichiers (backtranslated.json, train_augmented.json, config_D_reverse).
            print("Préparation partagée de D (modèle inverse + back-traduction + augmentation)...")
            run([sys.executable, str(TRAIN_SCRIPT["D"]), "--prepare-only"],
                log_path=logs_dir / "prepare_D_shared.log")
            print(f"Préparation D terminée (log: {logs_dir / 'prepare_D_shared.log'}).")

        jobs = [(config, seed) for config in args.configs for seed in args.seeds]
        print(f"{len(jobs)} job(s) d'entraînement répartis sur {len(gpu_ids)} GPU: {gpu_ids}")
        run_jobs_on_gpu_pool(jobs, gpu_ids, logs_dir)

    # Évaluation: rapide, un seul process à la fois suffit (I/O + génération courte).
    # IMPORTANT: evaluate_test.py réécrit tout evaluation_test_seedN.json à chaque appel
    # (pas de fusion) — tous les configs d'un même seed doivent donc être évalués en UNE
    # seule invocation, sinon le dernier appel écrase les scores des configs précédents.
    for seed in args.seeds:
        run([sys.executable, str(ROOT / "evaluate_test.py"), "--configs", *args.configs, "--seed", str(seed)])

    results_dir = ROOT / "results"

    # Le seed 42 (protocole de référence du papier) est souvent déjà entraîné/évalué
    # en amont (ex. étapes 05-07 de run_full_pipeline.sh) — on le réutilise dans
    # l'agrégat sans le réentraîner ni le réévaluer, plutôt que de dupliquer le run.
    effective_seeds = list(args.seeds)
    if 42 not in effective_seeds and (results_dir / "evaluation_test.json").exists():
        effective_seeds = [42] + effective_seeds
        print("Seed 42 trouvé dans results/evaluation_test.json — inclus dans l'agrégat "
              "sans réentraînement (déjà couvert par un run précédent).")

    def score_path_for(seed: int) -> Path:
        return results_dir / ("evaluation_test.json" if seed == 42 else f"evaluation_test_seed{seed}.json")

    summary_rows = []
    per_config: dict[str, dict[str, list[float]]] = {}
    for config in args.configs:
        per_config[config] = {m: [] for m in METRICS}
        for seed in effective_seeds:
            scores = json.loads(score_path_for(seed).read_text(encoding="utf-8"))[config]
            row = {"config": config, "seed": seed, **{m: scores[m] for m in METRICS}}
            summary_rows.append(row)
            for m in METRICS:
                per_config[config][m].append(scores[m])

    aggregate_rows = []
    for config in args.configs:
        row = {"config": config, "n_seeds": len(effective_seeds)}
        for m in METRICS:
            values = per_config[config][m]
            row[f"{m}_mean"] = round(mean(values), 4)
            row[f"{m}_std"] = round(pstdev(values), 4) if len(values) > 1 else 0.0
        aggregate_rows.append(row)

    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "multiseed_runs.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["config", "seed", *METRICS])
        writer.writeheader()
        writer.writerows(summary_rows)

    aggregate_fields = ["config", "n_seeds"] + [f"{m}_{stat}" for m in METRICS for stat in ("mean", "std")]
    with open(results_dir / "multiseed_summary.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=aggregate_fields)
        writer.writeheader()
        writer.writerows(aggregate_rows)

    print("\n=== Résumé multi-seed (moyenne ± écart-type) ===")
    for row in aggregate_rows:
        stats = ", ".join(f"{m}={row[f'{m}_mean']}±{row[f'{m}_std']}" for m in METRICS)
        print(f"Config {row['config']} (n={row['n_seeds']} seeds): {stats}")
    print(f"\nDétail par run -> {results_dir / 'multiseed_runs.csv'}")
    print(f"Agrégat -> {results_dir / 'multiseed_summary.csv'}")
    print(f"Logs d'entraînement -> {logs_dir}")


if __name__ == "__main__":
    main()

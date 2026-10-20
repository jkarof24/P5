#!/bin/bash
# Run the full end-to-end pipeline on AI-LAB via Slurm (sbatch).
#
# Resource notes for this workload (see --mem below):
#   * The full PyPI build peaks at 30-50 GB RAM -> --mem=64G (max 12h job).
#   * Heavy DuckDB edge joins benefit from many CPUs -> --cpus-per-task=16.
#   * torch is GPU-capable; 1 GPU stays well within the 4-GPU/job limit.
#   * .venv-p5 (Python 3.13, GPU torch) lives on the login node; Slurm jobs
#     run on the shared filesystem, so it is visible from the compute node.
#
# About --mem (AI-LAB: "--mem=24G ... Max 24GB per GPU"):
#   This script assumes the STANDARD Slurm meaning: --mem is the job's TOTAL
#   memory (system RAM), and "24GB per GPU" is the separate GPU-VRAM cap.
#   Under that reading --mem=64G hands the job 64 GB of RAM to cover the
#   30-50 GB PyPI build peak, and 1 GPU is plenty (the models are tiny).
#   If AI-LAB instead treats --mem as PER-GPU (hard-capped at 24 GB), this job
#   would be rejected: reach 30-50 GB by requesting multiple GPUs, e.g.
#       --gres=gpu:3 --mem=24G --time=10:00:00   # ~72 GB total
#
# Submit with:
#     sbatch run_ailab.sh
#
# Then check:
#     squeue -u $(whoami)          # job status
#     sbcat  -o ailab_full_run.out # view output (or: cat ailab_full_run.out)
set -euo pipefail
cd "$(dirname "$0")"

#SBATCH --job-name=ailab_full_run
#SBATCH --output=ailab_full_run.out
#SBATCH --error=ailab_full_run.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --partition=general

source .venv-p5/bin/activate
echo "==> Launching full pipeline (build -> stats -> train -> consolidate)"
bash run_all.sh

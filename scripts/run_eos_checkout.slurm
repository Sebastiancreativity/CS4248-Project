#!/usr/bin/env bash
#SBATCH --job-name=cs4248-eos
#SBATCH --partition=gpu-long
#SBATCH --gpus=a100-40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-eos-%A_%a.log
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through sbatch}"
cd "${SLURM_SUBMIT_DIR:?Submit from the new checkout root}"
python3 -B scripts/prepare_eos_checkout.py --verify
exec bash scripts/run_eos_revision.slurm "$@"

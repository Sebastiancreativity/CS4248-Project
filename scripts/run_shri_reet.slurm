#!/usr/bin/env bash
# Submit from ~/cs4248-mt: sbatch scripts/run_shri_reet.slurm
# Override partition/time/GPU using sbatch options if needed; inspect sinfo first.
#SBATCH --job-name=cs4248-shri-reet
#SBATCH --partition=gpu-long
#SBATCH --gpus=a100-40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-shri-reet-%j.log

set -euo pipefail
: "${SLURM_SUBMIT_DIR:?Submit from the project root}"
exec bash "$SLURM_SUBMIT_DIR/scripts/run_team.slurm" Shri Reet

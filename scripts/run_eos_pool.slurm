#!/usr/bin/env bash
#SBATCH --job-name=cs4248-eos-pool
#SBATCH --partition=gpu-long
#SBATCH --gpus=a100-40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-eos-pool-%A_%a.log
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through sbatch}"
cd "${SLURM_SUBMIT_DIR:?Submit from the prepared checkout root}"
revision="${1:?Missing evaluation revision}"
[[ "$revision" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || exit 1
if [[ -f eos_checkout.json ]]; then python3 -B scripts/prepare_eos_checkout.py --verify; fi
export CS4248_CLUSTER_DIR="${CS4248_CLUSTER_DIR:-$HOME/cs4248-mt-runtime}"
[[ "$CS4248_CLUSTER_DIR" = /* ]] || exit 1
export CS4248_MODEL_CACHE="$CS4248_CLUSTER_DIR/models"
export HF_HOME="$CS4248_MODEL_CACHE" HF_HUB_CACHE="$CS4248_MODEL_CACHE/hub"
export HF_DATASETS_CACHE="$CS4248_MODEL_CACHE/datasets" TORCH_HOME="$CS4248_MODEL_CACHE/torch"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TMPDIR
TMPDIR=$(mktemp -d /tmp/cs4248-eos-pool.XXXXXXXX)
trap 'rm -rf -- "$TMPDIR"' EXIT
exec 9<>"$CS4248_CLUSTER_DIR/finetune-environment.lock"
flock -s 9
source "$CS4248_CLUSTER_DIR/venv-finetune/bin/activate"
srun python -u scripts/eos_pool.py run --revision-dir "$PWD/artifacts/finetune/$revision"

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
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
action="${1:?Missing action}"
stage="${2:?Missing stage}"
source_run="${3:?Missing original training run}"
revision="${4:?Missing evaluation revision}"
case "$action" in prepare|seed|init|generate|score|collect|priority) ;; *) exit 1 ;; esac
case "$stage" in dev|test) ;; *) exit 1 ;; esac
[[ "$source_run" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ && "$revision" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ && "$source_run" != "$revision" ]] || exit 1
export CS4248_CLUSTER_DIR="${CS4248_CLUSTER_DIR:-$HOME/cs4248-mt-runtime}"
[[ "$CS4248_CLUSTER_DIR" = /* ]] || exit 1
export CS4248_MODEL_CACHE="$CS4248_CLUSTER_DIR/models"
export HF_HOME="$CS4248_MODEL_CACHE" HF_HUB_CACHE="$CS4248_MODEL_CACHE/hub"
export HF_DATASETS_CACHE="$CS4248_MODEL_CACHE/datasets" TORCH_HOME="$CS4248_MODEL_CACHE/torch"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TMPDIR
TMPDIR=$(mktemp -d /tmp/cs4248-eos.XXXXXXXX)
trap 'rm -rf -- "$TMPDIR"' EXIT
exec 9<>"$CS4248_CLUSTER_DIR/finetune-environment.lock"
flock -s 9
source "$CS4248_CLUSTER_DIR/venv-finetune/bin/activate"
args=("$action" --stage "$stage" --revision-dir "$PWD/artifacts/finetune/$revision")
[[ "$action" != prepare ]] || args+=(--source-run "$PWD/artifacts/finetune/$source_run")
if [[ "$action" == generate || "$action" == score ]]; then
    args+=(--index "${SLURM_ARRAY_TASK_ID:?Worker requires an array index}")
fi
srun python -u scripts/eos_revision.py "${args[@]}"

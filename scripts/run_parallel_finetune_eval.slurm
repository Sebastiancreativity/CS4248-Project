#!/usr/bin/env bash
#SBATCH --job-name=cs4248-eval
#SBATCH --partition=gpu-long
#SBATCH --gpus=a100-40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-ft-eval-%A_%a.log

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through sbatch}"
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
action="${1:?Expected init, generate, score or collect}"
stage="${2:?Expected dev or test}"
run_name="${3:-ft702010_v1}"
[[ "$run_name" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || exit 1
case "$action" in init|generate|score|collect) ;; *) exit 1 ;; esac
case "$stage" in dev|test) ;; *) exit 1 ;; esac
export CS4248_CLUSTER_DIR="${CS4248_CLUSTER_DIR:-$HOME/cs4248-mt-runtime}"
[[ "$CS4248_CLUSTER_DIR" = /* ]] || exit 1
export CS4248_MODEL_CACHE="$CS4248_CLUSTER_DIR/models"
export HF_HOME="$CS4248_MODEL_CACHE"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$HF_HOME/torch"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

# Multiprocessing sockets/temporary files belong on node-local storage, not NFS.
export TMPDIR
TMPDIR=$(mktemp -d /tmp/cs4248-eval.XXXXXXXX)
trap 'rm -rf -- "$TMPDIR"' EXIT
exec 9<>"$CS4248_CLUSTER_DIR/finetune-environment.lock"
flock -s 9
source "$CS4248_CLUSTER_DIR/venv-finetune/bin/activate"
args=("$action" --stage "$stage" --run-dir "$PWD/artifacts/finetune/$run_name")
if [[ "$action" == generate || "$action" == score ]]; then
    args+=(--index "${SLURM_ARRAY_TASK_ID:?Workers require an array index}")
fi
echo "Parallel evaluation: $action $stage; task=${SLURM_ARRAY_TASK_ID:-none}; node=$(hostname)"
srun python -u scripts/parallel_finetune_eval.py "${args[@]}"

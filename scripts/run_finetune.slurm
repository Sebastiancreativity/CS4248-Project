#!/usr/bin/env bash
# Used by submit_finetune_pipeline.sh; also accepts setup/train/dev/test for restarts.
#SBATCH --job-name=cs4248-ft
#SBATCH --partition=gpu-long
#SBATCH --gpus=a100-40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-finetune-%A_%a.log

set -euo pipefail
: "${SLURM_JOB_ID:?Submit using sbatch on the cluster}"
: "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
cd "$SLURM_SUBMIT_DIR"
[[ -f scripts/finetune_model.py && -f requirements-finetune.txt ]] || {
    echo 'Submit from the CS4248-Project root.' >&2; exit 1;
}
stage="${1:?Expected setup, train, dev or test}"
run_name="${2:-ft702010_v1}"
[[ "$run_name" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || { echo 'Invalid run name' >&2; exit 1; }
case "$stage" in setup|train|dev|test) ;; *) echo 'Unknown stage' >&2; exit 1 ;; esac
run_dir="$PWD/artifacts/finetune/$run_name"
export CS4248_CLUSTER_DIR="${CS4248_CLUSTER_DIR:-$HOME/cs4248-mt-runtime}"
[[ "$CS4248_CLUSTER_DIR" = /* ]] || { echo 'CS4248_CLUSTER_DIR must be absolute' >&2; exit 1; }
export CS4248_MODEL_CACHE="$CS4248_CLUSTER_DIR/models"
export HF_HOME="$CS4248_MODEL_CACHE"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$HF_HOME/torch"
export TMPDIR="$CS4248_CLUSTER_DIR/tmp"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR" "$CS4248_MODEL_CACHE" "$run_dir" logs

# A separate environment leaves existing inference jobs' dependencies alone.
ft_env="$CS4248_CLUSTER_DIR/venv-finetune"
exec 9>"$CS4248_CLUSTER_DIR/finetune-environment.lock"
if [[ "$stage" == setup ]]; then
    flock -x 9
    python3 -c 'import sys; assert (3, 10) <= sys.version_info[:2] <= (3, 12), "Use Python 3.10-3.12"'
    [[ -x "$ft_env/bin/python" ]] || python3 -m venv "$ft_env"
    source "$ft_env/bin/activate"
    python -m pip install --no-cache-dir -r requirements-finetune.txt
    python -m pip check
else
    # Keep dependencies fixed while any training/evaluation job is using them.
    flock -s 9
    [[ -x "$ft_env/bin/python" ]] || { echo 'Run setup first.' >&2; exit 1; }
    source "$ft_env/bin/activate"
fi
python - <<'PY'
import torch, transformers, peft
assert torch.cuda.is_available(), 'CUDA unavailable in the allocated job'
assert torch.cuda.is_bf16_supported(), 'Use an A100/H100-class GPU with BF16 support'
print('GPU:', torch.cuda.get_device_name(0), 'torch:', torch.__version__,
      'transformers:', transformers.__version__, 'peft:', peft.__version__)
PY
python -m pip freeze > "logs/finetune-${SLURM_JOB_ID}-environment.txt"
case "$stage" in
    setup)
        python scripts/finetune_pipeline.py prepare --run-dir "$run_dir" \
            --epochs "${CS4248_FT_EPOCHS:-3}" --learning-rate "${CS4248_FT_LR:-0.0001}"
        ;;
    train)
        model_index="${SLURM_ARRAY_TASK_ID:-${3:-}}"
        [[ "$model_index" =~ ^[0-7]$ ]] || { echo 'Training requires model index 0-7 or an array task.' >&2; exit 1; }
        srun python -u scripts/finetune_model.py --run-dir "$run_dir" --model-index "$model_index"
        ;;
    dev|test)
        srun python -u scripts/finetune_pipeline.py "$stage" --run-dir "$run_dir"
        ;;
esac

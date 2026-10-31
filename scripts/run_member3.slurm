#!/usr/bin/env bash
# Submit from ~/cs4248-mt: sbatch scripts/run_member3.slurm
# Override partition/time/GPU using sbatch options if needed; inspect sinfo first.
#SBATCH --job-name=cs4248-member3
#SBATCH --partition=gpu-long
#SBATCH --gpus=a100-40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-member3-%j.log

set -euo pipefail

# Refuse direct execution on the laptop or a login node.
: "${SLURM_JOB_ID:?Submit this script using sbatch on the cluster}"
: "${SLURM_SUBMIT_DIR:?Submit from the project root}"
# Override this if the home-directory quota cannot hold the checkpoints.
export CS4248_CLUSTER_DIR="${CS4248_CLUSTER_DIR:-$HOME/cs4248-mt-runtime}"
[[ "$CS4248_CLUSTER_DIR" = /* ]] || { echo "CS4248_CLUSTER_DIR must be absolute" >&2; exit 1; }
cd "$SLURM_SUBMIT_DIR"
[[ -f combinations.json && -f requirements-cluster.txt ]] || {
    echo "Submit from the CS4248-Project root directory" >&2; exit 1;
}

# All environments, downloads and temporary files stay on cluster storage.
export CS4248_MODEL_CACHE="$CS4248_CLUSTER_DIR/models"
export HF_HOME="$CS4248_MODEL_CACHE"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$HF_HOME/torch"
export TMPDIR="$CS4248_CLUSTER_DIR/tmp"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
mkdir -p "$CS4248_MODEL_CACHE" "$TMPDIR"

# Load your site's Python module before sbatch if python3 is not Python 3.10-3.12.
python3 -c 'import sys; assert (3, 10) <= sys.version_info[:2] <= (3, 12), "Use Python 3.10-3.12"'
if [[ ! -x "$CS4248_CLUSTER_DIR/venv/bin/python" ]]; then
    python3 -m venv "$CS4248_CLUSTER_DIR/venv"
fi
source "$CS4248_CLUSTER_DIR/venv/bin/activate"
python -m pip install --no-cache-dir -r requirements-cluster.txt
python -m pip check

# Check the allocated GPU before downloading any model weights.
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA unavailable: check the allocated GPU and NVIDIA driver"
assert torch.cuda.get_device_capability()[0] >= 8, "Use an A100/H100-class GPU for BF16"
x = torch.ones((32, 32), device="cuda", dtype=torch.bfloat16)
_ = x @ x
print("GPU:", torch.cuda.get_device_name(0), "PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
PY
mkdir -p logs
python -m pip freeze > "logs/member3-${SLURM_JOB_ID}-environment.txt"

# BLEU and chrF++ are automatic; these flags add COMET-22 and BERTScore.
# Both selectors and both development sets are explicit for reproducibility.
srun python -u scripts/run_ensembles.py \
    --combinations combinations.json \
    --owner "P Shricharan" \
    --dev dev_wmt dev_short \
    --selector both \
    --comet \
    --bertscore \
    --bertscore-device cuda \
    --bertscore-batch-size 16

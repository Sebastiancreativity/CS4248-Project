#!/usr/bin/env bash
# Submit setup -> eight model training tasks -> all dev comparisons -> all tests.
set -euo pipefail
[[ -f scripts/run_finetune.slurm && -f combinations_team_manifest.json ]] || {
    echo 'Run this from the CS4248-Project root.' >&2; exit 1;
}
run_name="${1:-ft702010_v1}"
[[ "$#" -le 1 && "$run_name" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || {
    echo 'Usage: bash scripts/submit_finetune_pipeline.sh [run_name]' >&2; exit 1;
}
parallel="${CS4248_FT_PARALLEL:-1}"
[[ "$parallel" =~ ^[1-8]$ ]] || { echo 'CS4248_FT_PARALLEL must be 1-8.' >&2; exit 1; }
command -v sbatch >/dev/null || { echo 'Run this on the Slurm cluster login node.' >&2; exit 1; }
# Optional environment overrides for a different GPU partition/allocation.
allocation=()
[[ -z "${CS4248_FT_PARTITION:-}" ]] || allocation+=(--partition="$CS4248_FT_PARTITION")
[[ -z "${CS4248_FT_GPUS:-}" ]] || allocation+=(--gpus="$CS4248_FT_GPUS")
[[ -z "${CS4248_FT_TIME:-}" ]] || allocation+=(--time="$CS4248_FT_TIME")
submit() {
    local response
    response=$(sbatch --parsable --kill-on-invalid-dep=yes "${allocation[@]}" "$@")
    printf '%s\n' "${response%%;*}"
}
setup_id=$(submit scripts/run_finetune.slurm setup "$run_name")
echo "Setup job: $setup_id"
train_id=$(submit --dependency="afterok:$setup_id" --array="0-7%$parallel" \
    scripts/run_finetune.slurm train "$run_name")
echo "Training array: $train_id (eight models; up to $parallel simultaneous GPU jobs)"
dev_id=$(submit --dependency="afterok:$train_id" scripts/run_finetune.slurm dev "$run_name")
echo "Dev evaluation: $dev_id (40 combinations x 3 selectors, before/after; freezes the winner)"
test_id=$(submit --dependency="afterok:$dev_id" scripts/run_finetune.slurm test "$run_name")
echo "Test evaluation: $test_id (internal 10%, WMT22 and Tatoeba)"
echo "Monitor: squeue -u $USER"
echo "Results: artifacts/finetune/$run_name/evaluation/"
echo "If a stage fails, dependent jobs cannot run. See FINETUNING.md for resume commands."

#!/usr/bin/env bash
# Resume evaluation only. Training and the original recipe are not changed.
set -euo pipefail
[[ -f scripts/run_parallel_finetune_eval.slurm ]] || { echo 'Run from the repository root.' >&2; exit 1; }
run_name="${1:-ft702010_v1}"
start="${2:-auto}"
[[ "$#" -le 2 && "$run_name" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || exit 1
case "$start" in auto|dev|test) ;; *) echo 'Stage must be auto, dev or test.' >&2; exit 1 ;; esac
parallel="${CS4248_EVAL_PARALLEL:-4}"
[[ "$parallel" =~ ^[1-8]$ ]] || { echo 'CS4248_EVAL_PARALLEL must be 1-8.' >&2; exit 1; }
[[ -f "artifacts/finetune/$run_name/run.json" ]] || { echo 'Prepared run not found.' >&2; exit 1; }
if [[ "$start" == auto ]]; then
    start=dev
    [[ ! -f "artifacts/finetune/$run_name/selection.json" ]] || start=test
fi
for command in sbatch scontrol scancel; do
    command -v "$command" >/dev/null || { echo 'Run this on the cluster login node.' >&2; exit 1; }
done
allocation=()
[[ -z "${CS4248_FT_PARTITION:-}" ]] || allocation+=(--partition="$CS4248_FT_PARTITION")
[[ -z "${CS4248_FT_GPUS:-}" ]] || allocation+=(--gpus="$CS4248_FT_GPUS")
[[ -z "${CS4248_FT_TIME:-}" ]] || allocation+=(--time="$CS4248_FT_TIME")
jobs=()
cleanup() {
    local result=$?
    if [[ "$result" -ne 0 && "${#jobs[@]}" -gt 0 ]]; then
        echo "Submission failed; cancelling this launch's jobs: ${jobs[*]}" >&2
        scancel "${jobs[@]}" || true
    fi
}
trap cleanup EXIT
submit() {
    local response
    response=$(sbatch --hold --parsable --kill-on-invalid-dep=yes "${allocation[@]}" "$@")
    job_id="${response%%;*}"
    [[ "$job_id" =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch response: $response" >&2; return 1; }
    jobs+=("$job_id")
}
dependency=()
stages=(test)
[[ "$start" != dev ]] || stages=(dev test)
for stage in "${stages[@]}"; do
    submit "${dependency[@]}" scripts/run_parallel_finetune_eval.slurm init "$stage" "$run_name"
    init_id="$job_id"
    submit --dependency="afterok:$init_id" --array="0-15%$parallel" \
        scripts/run_parallel_finetune_eval.slurm generate "$stage" "$run_name"
    generation_id="$job_id"
    submit --dependency="afterok:$generation_id" --array="0-9%$parallel" \
        scripts/run_parallel_finetune_eval.slurm score "$stage" "$run_name"
    scoring_id="$job_id"
    submit --dependency="afterok:$scoring_id" scripts/run_parallel_finetune_eval.slurm collect "$stage" "$run_name"
    dependency=(--dependency="afterok:$job_id")
    echo "$stage: prepare=$init_id generation=$generation_id scoring=$scoring_id collect=$job_id"
done
# Nothing runs until every dependency has been submitted successfully.
for job_id in "${jobs[@]}"; do
    scontrol release "$job_id"
done
trap - EXIT
echo "Submitted up to $parallel simultaneous GPU workers, subject to cluster limits."
echo "Jobs: ${jobs[*]}"
echo "Progress: python3 scripts/parallel_finetune_eval.py status --run-dir artifacts/finetune/$run_name"
echo 'Logs: slurm-ft-eval-<array-id>_<task-id>.log'

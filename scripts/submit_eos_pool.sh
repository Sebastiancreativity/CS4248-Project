#!/usr/bin/env bash
# Queue at most four reusable GPU jobs, not every model/metric task at once.
set -euo pipefail
source_run="${1:?Usage: $0 training_run revision_name dev_job_id pending_test_job_id}"
revision="${2:?Missing revision name}"
dev_job="${3:?Missing original dev job}"
old_test="${4:?Missing pending old test job}"
[[ "$#" -eq 4 && "$source_run" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ && "$revision" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || exit 1
[[ "$source_run" != "$revision" && "$dev_job" =~ ^[0-9]+$ && "$old_test" =~ ^[0-9]+$ && "$dev_job" != "$old_test" ]] || exit 1
[[ -f scripts/run_eos_pool.slurm && -f "artifacts/finetune/$source_run/run.json" ]] || { echo 'Run from the prepared checkout.' >&2; exit 1; }
workers="${CS4248_EVAL_PARALLEL:-4}"
[[ "$workers" =~ ^[1-4]$ ]] || { echo 'CS4248_EVAL_PARALLEL must be 1-4.' >&2; exit 1; }
python3 -B scripts/eos_pool.py init --source-run "$PWD/artifacts/finetune/$source_run" \
    --revision-dir "$PWD/artifacts/finetune/$revision" --dev-job "$dev_job"
pool_dir="$PWD/artifacts/finetune/.eos-pool-$revision"
exec 8<>"$pool_dir/submission.lock"
flock -n 8 || { echo 'Another submission is already in progress.' >&2; exit 1; }
record=$(scontrol -o show job "$old_test")
state=""; reason=""; dependency=""
for field in $record; do
    case "$field" in JobState=*) state="${field#*=}" ;; Reason=*) reason="${field#*=}" ;; Dependency=*) dependency="${field#*=}" ;; esac
done
[[ "$state" == PENDING && "$reason" != JobHeldAdmin ]] || { echo 'The old test must still be pending. Nothing was submitted.' >&2; exit 1; }
dependency="${dependency//\(unfulfilled\)/}"
dependency="${dependency//\(fulfilled\)/}"
[[ "$dependency" == "afterok:$dev_job" || "$dependency" == '(null)' ]] || { echo 'Unexpected old test dependency. Nothing submitted.' >&2; exit 1; }
already_held=false
[[ "$reason" != JobHeldUser ]] || already_held=true
pool_job=""
cleanup() {
    local result=$?
    if [[ "$result" -ne 0 ]]; then
        [[ -z "$pool_job" ]] || scancel "$pool_job" || true
        if [[ "$already_held" == false ]]; then scontrol release "$old_test" || true; fi
        echo 'Submission stopped; no running dev job was changed. The original test hold state was preserved.' >&2
    fi
}
scontrol hold "$old_test"
trap cleanup EXIT
record=$(scontrol -o show job "$old_test")
[[ " $record " == *" JobState=PENDING "* ]] || { echo 'Old test started during handover; stopping.' >&2; exit 1; }
allocation=()
[[ -z "${CS4248_FT_PARTITION:-}" ]] || allocation+=(--partition="$CS4248_FT_PARTITION")
[[ -z "${CS4248_FT_GPUS:-}" ]] || allocation+=(--gpus="$CS4248_FT_GPUS")
[[ -z "${CS4248_FT_TIME:-}" ]] || allocation+=(--time="$CS4248_FT_TIME")
while (( workers >= 1 )); do
    if response=$(sbatch --hold --parsable "${allocation[@]}" --array="0-$((workers-1))%$workers" \
        scripts/run_eos_pool.slurm "$revision" 2>"$pool_dir/submission-error.log"); then
        pool_job="${response%%;*}"
        [[ "$pool_job" =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch response: $response" >&2; pool_job=""; exit 1; }
        break
    fi
    cat "$pool_dir/submission-error.log" >&2
    if ! grep -Eq 'AssocMaxSubmitJobLimit|QOSMaxSubmitJobPerUserLimit|QOSMaxSubmitJobPerAccountLimit|QOSGrpSubmitJobsLimit' "$pool_dir/submission-error.log"; then
        exit 1
    fi
    workers=$((workers-1))
    (( workers < 1 )) || echo "Trying a smaller pool: $workers worker(s)."
done
[[ -n "$pool_job" ]] || { echo 'No submission slot is currently available; original jobs remain intact.' >&2; exit 1; }
record=$(scontrol -o show job "$old_test")
[[ " $record " == *" JobState=PENDING "* ]] || { echo 'Old test is no longer pending; refusing to cancel running work.' >&2; exit 1; }
scancel "$old_test"
trap - EXIT
printf '%s\n' "$pool_job" > "$pool_dir/submitted-job.txt"
echo "Original dev $dev_job continues. Replaced pending test $old_test with pool $pool_job ($workers workers)."
if ! scontrol release "$pool_job"; then
    echo "Pool is queued; retry: scontrol release $pool_job" >&2
    exit 1
fi
echo "Progress: python3 scripts/eos_pool.py status --revision-dir artifacts/finetune/$revision"
echo "Logs: slurm-eos-pool-${pool_job}_<worker>.log"

#!/usr/bin/env bash
# Insert report results between an existing dev job and its pending full test job.
set -euo pipefail
run_name="${1:?Usage: $0 run_name dev_job_id full_test_job_id}"
dev_job="${2:?Missing dev job ID}"
full_job="${3:?Missing pending full test job ID}"
[[ "$#" -eq 3 && "$run_name" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ && "$dev_job" =~ ^[0-9]+$ && "$full_job" =~ ^[0-9]+$ ]] || exit 1
[[ -f scripts/priority_test.py && -f "artifacts/finetune/$run_name/run.json" ]] || { echo 'Run from the prepared repository root.' >&2; exit 1; }
record=$(scontrol -o show job "$full_job")
state=""; dependency=""; reason=""
for field in $record; do
    case "$field" in JobState=*) state="${field#*=}" ;; Dependency=*) dependency="${field#*=}" ;; Reason=*) reason="${field#*=}" ;; esac
done
[[ "$state" == PENDING && "$reason" != JobHeldUser && "$reason" != JobHeldAdmin ]] || {
    echo 'The full test job must be pending and not already held. Nothing changed.' >&2; exit 1;
}
dependency="${dependency//\(unfulfilled\)/}"
dependency="${dependency//\(fulfilled\)/}"
[[ "$dependency" == "afterok:$dev_job" ]] || {
    echo "Expected the test job to depend only on dev job $dev_job; found $dependency. Nothing changed." >&2; exit 1;
}
allocation=()
[[ -z "${CS4248_FT_PARTITION:-}" ]] || allocation+=(--partition="$CS4248_FT_PARTITION")
[[ -z "${CS4248_FT_GPUS:-}" ]] || allocation+=(--gpus="$CS4248_FT_GPUS")
[[ -z "${CS4248_FT_TIME:-}" ]] || allocation+=(--time="$CS4248_FT_TIME")
priority_job=""
rollback() {
    local result=$?
    if [[ "$result" -ne 0 ]]; then
        echo 'Priority submission failed; restoring the original test dependency.' >&2
        if scontrol update "JobId=$full_job" "Dependency=afterok:$dev_job"; then
            scontrol release "$full_job" || true
            [[ -z "$priority_job" ]] || scancel "$priority_job" || true
        else
            echo "Restore failed. Inspect full test job $full_job and priority job ${priority_job:-not-submitted} before resubmitting." >&2
        fi
    fi
}
# Hold the pending test job so it cannot start while its dependency changes.
scontrol hold "$full_job"
trap rollback EXIT
record=$(scontrol -o show job "$full_job")
if [[ " $record " != *" JobState=PENDING "* ]]; then
    trap - EXIT
    scontrol release "$full_job" || true
    echo 'The full test job started during submission. No new job was submitted; inspect the queue.' >&2
    exit 1
fi
response=$(sbatch --hold --parsable --kill-on-invalid-dep=yes "${allocation[@]}" \
    --dependency="afterok:$dev_job" scripts/run_priority_test.slurm "$run_name")
priority_job="${response%%;*}"
[[ "$priority_job" =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch response: $response" >&2; priority_job=""; exit 1; }
scontrol update "JobId=$full_job" "Dependency=afterok:$priority_job"
# The new chain is complete. If a release fails, preserve it for manual release.
trap - EXIT
echo "Dev $dev_job -> priority report $priority_job -> full test $full_job"
if ! scontrol release "$priority_job" "$full_job"; then
    echo "Release failed. Retry: scontrol release $priority_job $full_job" >&2
    exit 1
fi
echo "No running job was cancelled. Log: slurm-priority-test-$priority_job.log"
echo "Report: artifacts/finetune/$run_name/evaluation/priority_test/report_scores.csv"

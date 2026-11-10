#!/usr/bin/env bash
# Replace a pending legacy test job; the original dev job keeps scoring.
set -euo pipefail
source_run="${1:?Usage: $0 training_run revision_name dev_job_id pending_test_job_id}"
revision="${2:?Missing revision name}"
dev_job="${3:?Missing running/completed dev job ID}"
old_test="${4:?Missing pending test job ID}"
[[ "$#" -eq 4 && "$source_run" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ && "$revision" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || exit 1
[[ "$source_run" != "$revision" && "$dev_job" =~ ^[0-9]+$ && "$old_test" =~ ^[0-9]+$ && "$dev_job" != "$old_test" ]] || exit 1
[[ -f scripts/eos_revision.py && -f scripts/run_eos_revision.slurm && -f "artifacts/finetune/$source_run/run.json" ]] || {
    echo 'Run from the prepared repository root after git pull.' >&2; exit 1;
}
[[ ! -e "artifacts/finetune/$revision" ]] || { echo 'Use a new revision name; do not submit duplicate workers.' >&2; exit 1; }
parallel="${CS4248_EVAL_PARALLEL:-4}"
[[ "$parallel" =~ ^[1-8]$ ]] || { echo 'CS4248_EVAL_PARALLEL must be 1-8.' >&2; exit 1; }
record=$(scontrol -o show job "$old_test")
state=""; reason=""; dependency=""
for field in $record; do
    case "$field" in JobState=*) state="${field#*=}" ;; Reason=*) reason="${field#*=}" ;; Dependency=*) dependency="${field#*=}" ;; esac
done
[[ "$state" == PENDING && "$reason" != JobHeldAdmin ]] || { echo 'The old test job must still be pending. Nothing changed.' >&2; exit 1; }
dependency="${dependency//\(unfulfilled\)/}"
dependency="${dependency//\(fulfilled\)/}"
[[ "$dependency" == "afterok:$dev_job" ]] || { echo "Unexpected old test dependency: $dependency. Nothing changed." >&2; exit 1; }
already_held=false
[[ "$reason" != JobHeldUser ]] || already_held=true
allocation=()
[[ -z "${CS4248_FT_PARTITION:-}" ]] || allocation+=(--partition="$CS4248_FT_PARTITION")
[[ -z "${CS4248_FT_GPUS:-}" ]] || allocation+=(--gpus="$CS4248_FT_GPUS")
[[ -z "${CS4248_FT_TIME:-}" ]] || allocation+=(--time="$CS4248_FT_TIME")
jobs=()
committed=false
cleanup() {
    local result=$?
    if [[ "$result" -ne 0 && "$committed" == false ]]; then
        [[ "${#jobs[@]}" -eq 0 ]] || scancel "${jobs[@]}" || true
        if [[ "$already_held" == false ]]; then scontrol release "$old_test" || true; fi
        echo 'Submission failed. New held jobs were cancelled; the original test hold state was restored.' >&2
    fi
}
scontrol hold "$old_test"
trap cleanup EXIT
record=$(scontrol -o show job "$old_test")
[[ " $record " == *" JobState=PENDING "* ]] || { echo 'The old test started during handover; no replacement will run.' >&2; exit 1; }
submit() {
    local response
    response=$(sbatch --hold --parsable --kill-on-invalid-dep=yes "${allocation[@]}" "$@")
    job_id="${response%%;*}"
    [[ "$job_id" =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch response: $response" >&2; exit 1; }
    jobs+=("$job_id")
}
launcher=scripts/run_eos_revision.slurm
# Validation can overlap the original scoring job: it reads only final raw
# translations/adapters and writes exclusively to the new revision directory.
submit "$launcher" prepare dev "$source_run" "$revision"
prepare_id="$job_id"
submit --dependency="afterok:$prepare_id" --array="5,13%$parallel" "$launcher" generate dev "$source_run" "$revision"
dev_generate="$job_id"
submit --dependency="afterok:$dev_generate:$dev_job" "$launcher" seed dev "$source_run" "$revision"
seed_id="$job_id"
submit --dependency="afterok:$seed_id" --array="0-9%$parallel" "$launcher" score dev "$source_run" "$revision"
dev_scores="$job_id"
submit --dependency="afterok:$dev_scores" "$launcher" collect dev "$source_run" "$revision"
dev_collect="$job_id"
submit --dependency="afterok:$dev_collect" "$launcher" init test "$source_run" "$revision"
test_init="$job_id"
submit --dependency="afterok:$test_init" --array="0-15%$parallel" "$launcher" generate test "$source_run" "$revision"
test_generate="$job_id"
# Generate the entire test candidate pool in parallel, then publish the four
# report systems first. The subsequent 720-row sweep reuses these candidates.
submit --dependency="afterok:$test_generate" "$launcher" priority test "$source_run" "$revision"
report_id="$job_id"
submit --dependency="afterok:$report_id" --array="0-9%$parallel" "$launcher" score test "$source_run" "$revision"
test_scores="$job_id"
submit --dependency="afterok:$test_scores" "$launcher" collect test "$source_run" "$revision"
test_collect="$job_id"
# Cancel only the pending obsolete test, after the full replacement is queued.
record=$(scontrol -o show job "$old_test")
[[ " $record " == *" JobState=PENDING "* ]] || { echo 'Old test no longer pending; refusing to cancel running work.' >&2; exit 1; }
scancel "$old_test"
committed=true
trap - EXIT
echo "Original dev $dev_job continues. Replaced pending test $old_test."
echo "EOS validation: prepare=$prepare_id generate=$dev_generate seed=$seed_id score=$dev_scores collect=$dev_collect"
echo "Corrected test: init=$test_init generate=$test_generate report=$report_id score=$test_scores collect=$test_collect"
echo "New jobs: ${jobs[*]}"
if ! scontrol release "${jobs[@]}"; then
    echo "Replacement is queued. Retry release: scontrol release ${jobs[*]}" >&2
    exit 1
fi
echo "Up to $parallel GPU workers per array, subject to cluster limits. No retraining."
echo "Progress: python3 scripts/eos_revision.py status --revision-dir artifacts/finetune/$revision"
echo "Report CSV: artifacts/finetune/$revision/evaluation/priority_test/report_scores.csv"
echo "Full CSV: artifacts/finetune/$revision/evaluation/test/scores_comparison.csv"

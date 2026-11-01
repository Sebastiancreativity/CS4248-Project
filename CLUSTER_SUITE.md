# All combinations and Member 4

Reviewed after pulling MT commit `0c5a04d` (Tower introduction cleaning fix).
No weights are downloaded locally. The original `combinations.json` still
contains Member 3's assigned ten triples.

## Experiments

1. `combinations_all.json`: all 56 distinct three-model combinations from the
   eight supported models, including combinations with multiple Qwen sizes.
   This is broader than the screenshot's 40 assigned combinations (16 extra).
   `--selector agreement` runs chrF++ and BERTScore selection on the same
   candidates. With two dev sets this produces 224 evaluation rows.
2. `combinations_member4.json`: H+T+A, H+T+M, H+T+N, H+A+M, H+A+N,
   H+M+N, T+A+M, T+A+N, T+M+N, A+M+N, in the screenshot's order.
   `--selector both` retains its original meaning: COMETKiwi and chrF++.
   With two dev sets this produces 40 evaluation rows.

Both stages report BLEU, chrF++, COMET-22 and BERTScore P/R/F1. BERTScore
selection uses the mean unrounded F1 against the two other candidates, never
the gold reference. It uses the same English DeBERTa encoder as evaluation,
without IDF or baseline rescaling. Empty peer comparisons contribute zero;
empty candidates cannot win against nonempty candidates; ties keep model order.
Treat BERTScore evaluation as potentially favouring the matching selector;
also compare COMET, overlap metrics, individual-model baselines and examples.
Choose systems on dev data, and reserve test data for final evaluation.

## Run

Upload updated scripts and the two new JSON files from laptop PowerShell:

```powershell
cd C:\Users\Shricharan\VSCodeProjects\CS4248-Project
scp -J shri@sjump.comp.nus.edu.sg -r scripts combinations_all.json combinations_member4.json requirements-cluster.txt shri@xlogin.comp.nus.edu.sg:/home/s/shri/cs4248-mt/
```

Then on the cluster:

```bash
cd ~/cs4248-mt
sbatch scripts/run_full_suite.slurm
```

The script requests an A100 40 GB, 8 CPUs, 64 GB RAM and 48 hours in `gpu-long`.
48 hours is a time allowance, not a runtime estimate. It uses the existing
`~/cs4248-mt-runtime` environment, model cache and Hugging Face credentials.
COMETKiwi still requires approved access. The two stages are attempted separately;
the job exits unsuccessfully if either stage fails. Do not submit concurrent
copies against the same output folders.

## Progress and restart

Model outputs are generated once per model/dev set. Cached raw outputs are
re-cleaned with the current cleaner before selection, including Tower's fix.
This can change scores relative to older reports even with the same selector;
compare the two new selectors within this refreshed run for a fair comparison.

Pairwise BERTScore vectors are cached per directed model pair/dev set and
invalidated when candidate text or the scorer hash changes. Each completed
evaluation is saved atomically as a checkpoint. Re-submitting the same command
reuses matching evaluations; code, package, configuration or evaluated-data
changes invalidate them. An interrupted generation for one dev set restarts
that generation. The final summaries are rebuilt from checkpointed results.

Output folders (original Member 3 reports are kept separately):

```text
results/ensembles/P Shricharan/all56_agreement/
results/ensembles/P Shricharan/member4_original/
```

Each folder contains `scores_comparison.csv`, per-selector CSV/JSON summaries,
and numbered experiment folders with hypotheses and checkpoints. The two
comparison CSVs total 264 rows. Experiment numbers belong to their own JSON
file; the all-56 experiment numbers do not match Member 3's numbering.

```bash
squeue -u "$USER"
tail -n 60 "$(ls -t slurm-suite-*.log | head -n 1)"
```

Local checks use mocked encoders and tiny fixtures. The full GPU run must be
performed on the cluster; no local GPU scoring or downloads were performed.

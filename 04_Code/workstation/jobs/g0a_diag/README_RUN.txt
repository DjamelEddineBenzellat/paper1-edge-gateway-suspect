JOB {JOB_ID}
DL-35 diagnostic after gate G0-a failed at eps = 8. NO gate decision.
It answers one question: is P-TA weak because of the privacy noise, or
because of our pipeline, model or metric?
- (a) P-TA and P-TG-c WITHOUT privacy noise (z = 0), seeds 0-2: 6 runs.
- (b) The eps = 8 pilot again (P-TA, P-TG-c, local-only), seeds 0-2: 9 runs,
  with extra measures (per-class AUROC, F1 with tuned thresholds) and the
  fold-9 scores saved.
Expected time on the GTX 1660 SUPER: about 1-1.5 hours [Estimate].

=== BEFORE ===
The setup job JOB_20261112_G0-03_workstation-setup must have finished with status OK
(environment ~/paper1/venvs/paper1-gpu and data in ~/paper1/PhD_data).

=== RUN ===
1. Unzip {JOB_ID}.zip into ~/paper1/job/
2. Open a terminal in ~/paper1/job/{JOB_ID}/
3. Run this ONE command (one line):
     python3.12 run_job.py --base-dir ~/paper1 --data-dir ~/paper1/PhD_data
   Keep the computer on. Do not close the terminal.

What you will see first (about 5 minutes):
- the unit tests ("OK");
- a "preflight": 1 round of every mode on the GPU, with the new options.
  If memory or setup is wrong, the job stops HERE, after minutes.

=== IF IT STOPS ===
- Power cut, or you closed it: run the SAME command again in the SAME folder.
  Finished runs are kept in the folder "state" and are skipped.
- "CUDA out of memory": the job already retries by itself with fewer
  gateways on the GPU at once (3 attempts). You do not need to do anything.

=== BRING BACK ===
At the end the last lines show "RESULTS ZIP: ...".
Bring back:  ~/paper1/job/{JOB_ID}/{JOB_ID}_results.zip
Put it on the laptop in:  05_Experiments/workstation/inbox/
Bring it back also if the job FAILED: the zip contains the error, the finished
runs and a partial summary.

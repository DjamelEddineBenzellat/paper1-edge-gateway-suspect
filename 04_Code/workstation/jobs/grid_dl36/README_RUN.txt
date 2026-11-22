JOB {JOB_ID}
DL-36: the eps = 8 grid again, scored with the new metric (macro-F1 with the
best threshold per class, chosen on fold 9). Needed before E1.
- 9 P-TA runs (3 learning rates x 3 C_ex, seed 0, eps = 8).
- 3 local-only runs (3 learning rates, seed 0).
- Then it chooses the default settings for E1 with the new metric, and
  shows the old choice next to the new one.
Expected time on the GTX 1660 SUPER: about 1 hour [Estimate].

=== BEFORE ===
The setup job JOB_20261112_G0-03_workstation-setup must have finished with status OK
(environment ~/paper1/venvs/paper1-gpu and data in ~/paper1/PhD_data).

=== RUN ===
1. Unzip {JOB_ID}.zip into ~/paper1/job/
2. Open a terminal in ~/paper1/job/{JOB_ID}/
3. Run this ONE command (one line):
     python3.12 run_job.py --base-dir ~/paper1 --data-dir ~/paper1/PhD_data
   Keep the computer on. Do not close the terminal.

What you will see first (about 2 minutes):
- the unit tests ("OK");
- a "preflight": 1 round of P-TA and 1 epoch of local-only on the GPU.
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
Bring it back also if the job FAILED: the zip contains the error and the finished runs.

JOB {JOB_ID}
Gate G0-a: the fold-9 grid and pilot (plan tasks G0-14 and G0-15).
- Stage eps = 3: grid (9 P-TA + 3 local-only), selection, pilot (3 P-TA + 3 local-only + 3 P-TG-c).
- Stage eps = 8 (only with --also-eps 8, DL-34): the same again at eps = 8.
  The local-only runs are copied, because they use no privacy.
- Finished runs from an earlier G0-a job are reused (--reuse-state) and not run again.
Expected time on the GTX 1660 SUPER for this job: about 1.5-2.5 hours [Estimate].

=== BEFORE ===
The setup job JOB_20261112_G0-03_workstation-setup must have finished with status OK
(environment ~/paper1/venvs/paper1-gpu and data in ~/paper1/PhD_data).

=== RUN ===
1. Unzip {JOB_ID}.zip into ~/paper1/job/
2. Open a terminal in ~/paper1/job/{JOB_ID}/
3. Run this ONE command (one line):
     python3.12 run_job.py --base-dir ~/paper1 --data-dir ~/paper1/PhD_data --reuse-state ~/paper1/job/JOB_20261115_G0-14_g0a-grid-pilot --also-eps 8
   Keep the computer on. Do not close the terminal.

What you will see first (about 5 minutes):
- the unit tests ("OK");
- a "preflight": 1 round of every mode on the GPU. If memory or setup is
  wrong, the job stops HERE, after minutes, not after hours.

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

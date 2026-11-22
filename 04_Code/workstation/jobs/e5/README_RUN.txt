JOB {JOB_ID}
E5 for Paper 1: gateway tampering (plan task X-9; DL-43 to DL-46, DL-48).
- One gateway tampers in each run: scaling x2, sign flip, injection,
  silent drop of half its inputs, or the adaptive gateway (DL-46: it
  tests each flip size against the cloud's check of the same round).
  Plus runs without tampering (the reference).
- 3 detectors: our cloud check (size-aware distance and rate alert, DL-48),
  HFLMLD native (P-PE without DP), and the HFLMLD cloud rule on P-TA.
- 180 runs (6 x 3 x 10 seeds) on the mixed partition (gateways 0-2 with
  alpha 0.1).
Every run writes a SEALED fold-10 file. They come back in
results/sealed_fold10/ and must NOT be opened (only by the opening
script, after the results are checked).
Expected time on the GTX 1660 SUPER: about 14-16 hours [Estimate].

=== BEFORE ===
- Start this job only AFTER the E4 job (JOB_20261119_X-9_e4-labelflip) has
  finished: one GPU, one job at a time.
- This job replaces JOB_20261119_X-9_e5 (old adaptive gateway). Do NOT run
  the old one.
- The workstation environment ~/paper1/venvs/paper1-gpu and the data in
  ~/paper1/PhD_data are the same as for the earlier jobs.
- Switch off automatic sleep / suspend on the workstation.

=== RUN ===
1. Unzip {JOB_ID}.zip into ~/paper1/job/
2. Open a terminal in ~/paper1/job/{JOB_ID}/
3. Run this ONE command (one line):
     python3.12 run_job.py --base-dir ~/paper1 --data-dir ~/paper1/PhD_data
   Keep the computer on. Do not close the terminal.

What you will see:
- the unit tests ("OK");
- the PREFLIGHT (about 15 minutes): 12 short runs and the line
  "preflight OK";
- the 180 runs. After each seed: "E5 seed N: done".
Normal messages you can ignore: "DEPRECATED FEATURE: run_simulation",
"FutureWarning ... RAY_ACCEL_ENV_VAR...", "Manually terminating ClientAppActor".

=== IF IT STOPS ===
- Power cut, or you closed it: run the SAME command again in the SAME folder.
  Finished runs are kept in "state_e5" and are skipped.
- "CUDA out of memory" or a Ray timeout: the job retries each run by itself
  (3 attempts). If a run still fails, it is skipped and the job goes on.
  At the end the status is then INCOMPLETE: run the SAME command again;
  it runs only the missing runs.

=== BRING BACK ===
At the end the last lines show "RESULTS ZIP: ..." and "Status: OK".
Bring back:  ~/paper1/job/{JOB_ID}/{JOB_ID}_results.zip
Put it on the laptop in:  05_Experiments/workstation/inbox/
Bring it back also if the status is FAILED or INCOMPLETE: the zip contains
the error and the finished runs.

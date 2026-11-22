JOB {JOB_ID}
DL-40: P-PE's cloud step (plan task M3-01).
- Part 1, grid on fold 9: 6 cloud steps x seeds 0-4 = 30 short P-PE runs.
  They score fold 9 only (no fold 10).
- The job then chooses the cloud step by the fixed DL-40 rule.
- Part 2: the 10 E1 P-PE runs again with that cloud step. Each also writes
  a SEALED fold-10 file. It comes back in results/sealed_fold10/ and must
  NOT be opened before gate G1 passes (DL-39).
Expected time on the GTX 1660 SUPER: about 3.5-4 hours [Estimate].

=== BEFORE ===
- The setup job JOB_20261112_G0-03_workstation-setup must have finished with
  status OK (environment ~/paper1/venvs/paper1-gpu, data in ~/paper1/PhD_data).
- Switch off automatic sleep / suspend on the workstation.

=== RUN ===
1. Unzip {JOB_ID}.zip into ~/paper1/job/
2. Open a terminal in ~/paper1/job/{JOB_ID}/
3. Run this ONE command (one line):
     python3.12 run_job.py --base-dir ~/paper1 --data-dir ~/paper1/PhD_data
   Keep the computer on. Do not close the terminal.

What you will see:
- the unit tests ("OK") and a short preflight run (about 1 minute);
- the 30 grid runs. After each seed: "grid seed N: all 6 cloud steps done";
- a small table of the 6 cloud steps and the line "SELECTED eta_c = ...";
- the 10 E1 P-PE runs.
Normal messages you can ignore: "DEPRECATED FEATURE: run_simulation",
"FutureWarning ... RAY_ACCEL_ENV_VAR...", "Manually terminating ClientAppActor".
If you see "DL-40 STOP RULE", that is not an error: the job ends early on
purpose. Bring the zip back as usual.

=== IF IT STOPS ===
- Power cut, or you closed it: run the SAME command again in the SAME folder.
  Finished runs are kept in "state_grid" and "state_e1" and are skipped.
- "CUDA out of memory": the job already retries by itself with fewer gateways
  on the GPU at once (3 attempts). You do not need to do anything.

=== BRING BACK ===
At the end the last lines show "RESULTS ZIP: ..." and "Status: OK".
Bring back:  ~/paper1/job/{JOB_ID}/{JOB_ID}_results.zip
Put it on the laptop in:  05_Experiments/workstation/inbox/
Bring it back also if the job FAILED: the zip contains the error and the
finished runs.

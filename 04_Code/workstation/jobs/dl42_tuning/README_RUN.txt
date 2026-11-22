JOB {JOB_ID}
DL-42: the same small tuning grid for all five DP designs (plan tasks X-1/X-2).
- Part 1, grid on fold 9: lr {0.01, 0.05, 0.1} x C_ex {0.5, 1.0, 2.0}
  x seeds 0-2 for P-TG, P-TG-c, P-PE, P-FR and P-TA = 126 runs, plus 3
  check runs. They score fold 9 only (no fold 10).
- The job then chooses each design's point by the fixed DL-42 rule.
- Part 2: for each design whose point changed, the 10 E1 seeds again at
  the new point (0 to 50 runs). Each also writes a SEALED fold-10 file.
  It comes back in results/sealed_fold10/ and must NOT be opened (only
  by the opening script, after the results are checked).
Expected time on the GTX 1660 SUPER: about 12-16 hours [Estimate]
(one night and the next day).

=== BEFORE ===
- The workstation environment ~/paper1/venvs/paper1-gpu and the data in
  ~/paper1/PhD_data are the same as for the E1 and DL-40 jobs.
- Switch off automatic sleep / suspend on the workstation.

=== RUN ===
1. Unzip {JOB_ID}.zip into ~/paper1/job/
2. Open a terminal in ~/paper1/job/{JOB_ID}/
3. Run this ONE command (one line):
     python3.12 run_job.py --base-dir ~/paper1 --data-dir ~/paper1/PhD_data
   Keep the computer on. Do not close the terminal.

What you will see:
- the unit tests ("OK");
- the PREFLIGHT (about 10 minutes): 5 short runs, a test selection,
  2 short runs with sealed files, and the line "preflight OK";
- the reuse check: 3 runs and lines "reuse check ...: MATCH"
  ("DIFFERENT" is not an error: the job then runs 2 extra runs for that
  design by itself);
- the 126 grid runs. After each seed: "grid seed N: done";
- a table of the grid points and, per design, "-> keep" or "-> change";
- the reruns of the designs that changed.
Normal messages you can ignore: "DEPRECATED FEATURE: run_simulation",
"FutureWarning ... RAY_ACCEL_ENV_VAR...", "Manually terminating ClientAppActor".

=== IF IT STOPS ===
- Power cut, or you closed it: run the SAME command again in the SAME folder.
  Finished runs are kept in "state_grid" and "state_rerun" and are skipped.
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

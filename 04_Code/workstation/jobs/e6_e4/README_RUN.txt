JOB {JOB_ID}
E6 and E4 for Paper 1 (plan task X-9; DL-41, DL-42, DL-43, DL-44).
- E4 positive control: the attacks on plain FedAvg (2 runs). They must
  work; otherwise the job does NOT start the E4 attack runs.
- E6: 15 norm runs, then 30 runs for the eps line and the k line.
- E4: 10 P-TA reference runs (no attack) and 150 attack runs
  (3 attacks x 5 designs x 10 seeds).
All E6 and E4 runs write a SEALED fold-10 file. They come back in
results/sealed_fold10/ and must NOT be opened (only by the opening
script, after the results are checked).
Expected time on the GTX 1660 SUPER: about 18-21 hours [Estimate]
(about one night, one day and part of the next night).

=== BEFORE ===
- The workstation environment ~/paper1/venvs/paper1-gpu and the data in
  ~/paper1/PhD_data are the same as for the E1, DL-40 and DL-42 jobs.
- Switch off automatic sleep / suspend on the workstation.

=== RUN ===
1. Unzip {JOB_ID}.zip into ~/paper1/job/
2. Open a terminal in ~/paper1/job/{JOB_ID}/
3. Run this ONE command (one line):
     python3.12 run_job.py --base-dir ~/paper1 --data-dir ~/paper1/PhD_data
   Keep the computer on. Do not close the terminal.

What you will see:
- the unit tests ("OK");
- the PREFLIGHT (about 20-25 minutes): 17 short runs of every type, and
  the line "preflight OK";
- the positive control: the line "positive control: ... PASS ... PASS";
- the E6 runs, the 10 reference runs, then the E4 runs. After each E4
  seed: "E4 seed N: done".
Normal messages you can ignore: "DEPRECATED FEATURE: run_simulation",
"FutureWarning ... RAY_ACCEL_ENV_VAR...", "Manually terminating ClientAppActor".

=== IF IT STOPS ===
- Power cut, or you closed it: run the SAME command again in the SAME folder.
  Finished runs are kept in the "state_..." folders and are skipped.
- "CUDA out of memory" or a Ray timeout: the job retries each run by itself
  (3 attempts). If a run still fails, it is skipped and the job goes on.
  At the end the status is then INCOMPLETE: run the SAME command again;
  it runs only the missing runs.
- Status PC_FAILED: the attacks did not work on FedAvg. This is a result,
  not an error. Bring the zip back; Claude prepares a DL row.

=== BRING BACK ===
At the end the last lines show "RESULTS ZIP: ..." and "Status: OK".
Bring back:  ~/paper1/job/{JOB_ID}/{JOB_ID}_results.zip
Put it on the laptop in:  05_Experiments/workstation/inbox/
Bring it back also if the status is FAILED, INCOMPLETE or PC_FAILED: the
zip contains the error and the finished runs.

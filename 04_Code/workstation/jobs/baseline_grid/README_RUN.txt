JOB {JOB_ID}
Baseline grid (DL-37, point 3): choose the learning rate of the 3 non-private
baselines on fold 9, with the new score (DL-36).
- centralized: lr 0.01 / 0.05 / 0.1, 100 epochs (scored at 10, 25, 50, 100)
- FedAvg:      lr 0.01 / 0.05 / 0.1, 100 rounds
- HierFAVG:    lr 0.01 / 0.05 / 0.1, 100 device rounds (20 cloud rounds x 5)
9 runs in total, seed 0, no privacy noise.

THIS TIME IT RUNS ON THE LAPTOP (CPU), because the workstation cannot be reached.
Expected time on the laptop: about 3-4 hours [Estimate].

=== BEFORE ===
- Plug in the laptop.
- Switch off automatic sleep / suspend for these hours.
- Close heavy programs (browser tabs with video, other big jobs).

=== RUN (copy each line into a terminal) ===
1. Make a folder for laptop jobs (once):
     mkdir -p <BASE_DIR>/jobs
2. Unzip the job into it:
     unzip <BASE_DIR>/PhD_Paper1_EdgeGatewaySuspect/05_Experiments/workstation/outbox/{JOB_ID}.zip -d <BASE_DIR>/jobs/
3. Go into the job folder:
     cd <BASE_DIR>/jobs/{JOB_ID}
4. Start the job (ONE line):
     python3 run_job.py --base-dir <BASE_DIR> --data-dir <BASE_DIR>/PhD_data --venv-python <BASE_DIR>/venvs/paper1/bin/python --device cpu
   Do not close the terminal while it runs.

What you will see first (about 5 minutes):
- the unit tests ("OK");
- a "preflight": one short run of each baseline. If something is wrong,
  the job stops HERE, after minutes, not after hours.
Normal messages you can ignore: "DEPRECATED FEATURE: run_simulation",
"FutureWarning ... RAY_ACCEL_ENV_VAR...", "Manually terminating ClientAppActor".

=== IF IT STOPS ===
Run the SAME command again in the SAME folder (step 3 and 4).
Finished runs are kept in the folder "state" and are skipped.

=== BRING BACK ===
At the end the last lines show "RESULTS ZIP: ..." and "Status: OK".
Copy the results zip into the project inbox:
     cp <BASE_DIR>/jobs/{JOB_ID}/{JOB_ID}_results.zip <BASE_DIR>/PhD_Paper1_EdgeGatewaySuspect/05_Experiments/workstation/inbox/
Do this also if the job FAILED: the zip contains the error and the finished runs.

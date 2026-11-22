JOB {JOB_ID}
Experiment E1: benign accuracy of the 9 designs (plan task M3-01).
- P-TG, P-TG-c, P-PE, P-FR, P-TA, centralized, local-only, FedAvg, HierFAVG.
- 10 seeds (0-9): 90 runs, one config file per run (configs/E1/).
- Each run also writes a SEALED fold-10 file. It comes back in
  results/sealed_fold10/ and must NOT be opened before gate G1 passes (DL-39).
Expected time on the GTX 1660 SUPER: about 7-8 hours [Estimate]. Start it in
the evening and let it run through the night.

=== BEFORE ===
- The setup job JOB_20261112_G0-03_workstation-setup must have finished with
  status OK (environment ~/paper1/venvs/paper1-gpu, data in ~/paper1/PhD_data).
- Switch off automatic sleep / suspend on the workstation for the night
  (on 15 Nov a 2.5-minute pause stopped one run once).

=== RUN ===
1. Unzip {JOB_ID}.zip into ~/paper1/job/
2. Open a terminal in ~/paper1/job/{JOB_ID}/
3. Run this ONE command (one line):
     python3.12 run_job.py --base-dir ~/paper1 --data-dir ~/paper1/PhD_data
   Keep the computer on. Do not close the terminal.

What you will see first (about 10 minutes):
- the unit tests ("OK");
- a "preflight": a short run of each of the 9 designs on the GPU. If memory
  or setup is wrong, the job stops HERE, after minutes, not after hours.
Then the 90 runs, seed by seed. After each seed you see
"seed N: all 9 configurations done".
Normal messages you can ignore: "DEPRECATED FEATURE: run_simulation",
"FutureWarning ... RAY_ACCEL_ENV_VAR...", "Manually terminating ClientAppActor".

=== IF IT STOPS ===
- Power cut, or you closed it: run the SAME command again in the SAME folder.
  Finished runs are kept in the folder "state" and are skipped.
- "CUDA out of memory": the job already retries by itself with fewer gateways
  on the GPU at once (3 attempts). You do not need to do anything.

=== BRING BACK ===
At the end the last lines show "RESULTS ZIP: ..." and "Status: OK".
Bring back:  ~/paper1/job/{JOB_ID}/{JOB_ID}_results.zip
Put it on the laptop in:  05_Experiments/workstation/inbox/
Bring it back also if the job FAILED: the zip contains the error, the finished
runs and a partial summary.

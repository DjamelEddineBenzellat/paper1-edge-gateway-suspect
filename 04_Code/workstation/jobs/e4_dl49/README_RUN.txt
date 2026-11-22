JOB {JOB_ID}
E4 for Paper 1: device poisoning by label flipping (plan task X-9; DL-49).
- Attackers flip the labels of their records: either about 10% of all
  devices (uniform), or 15 of the 30 devices of one gateway (concentrated).
- 100 runs: 2 attacks x 5 designs x 10 seeds.
- FIRST a positive control on plain FedAvg (about 7 minutes). If the attack
  does not damage plain FedAvg enough, the job STOPS at once (status
  PC_FAILED) and no other run starts.
Every E4 run writes a SEALED fold-10 file. They come back in
results/sealed_fold10/ and must NOT be opened (only by the opening
script, after the results are checked).
Expected time on the GTX 1660 SUPER: about 10 hours [Estimate].

=== BEFORE ===
- One GPU, one job at a time: no other job may run on the workstation.
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
- the PREFLIGHT (about 10 minutes): 12 short runs, then "preflight OK";
- the positive control: the line
  "positive control (DL-49): labelflip drop ... PASS; concentrated_labelflip
  drop ... PASS";
- the E4 runs. After each seed: "E4 seed N: done".
Normal messages you can ignore: "DEPRECATED FEATURE: run_simulation",
"FutureWarning ... RAY_ACCEL_ENV_VAR...", "Manually terminating ClientAppActor".

=== IF IT STOPS ===
- Power cut, or you closed it: run the SAME command again in the SAME folder.
  Finished runs are kept in the "state_..." folders and are skipped.
- "CUDA out of memory" or a Ray timeout: the job retries each run by itself
  (3 attempts). If a run still fails, it is skipped and the job goes on.

=== AFTER ===
The job ALWAYS writes {JOB_ID}_results.zip in this folder, also if it fails.
Copy it to the laptop: 05_Experiments/workstation/inbox/
The last line shows the status:
- OK          : all 100 runs are done.
- PC_FAILED   : the positive control failed after about 20 minutes; no E4
                run was started. Bring the zip back.
- INCOMPLETE  : some runs failed 3 times; run the same command again.
- FAILED      : bring the zip back; it contains the error.

JOB {JOB_ID}
The E4 gate for Paper 1 (plan task X-9; DL-47).
- It runs the attack "positive control" on plain FedAvg at full length
  (100 rounds) for 13 attack strengths: 8 backdoor and 5 label-flip runs.
- A rule written in DL-47 then chooses the attack strength for E4.
- No run writes a fold-10 file. Nothing in this job is sealed.
- Only if this gate passes does Claude build the long E4 job.
Expected time on the GTX 1660 SUPER: about 50-60 minutes [Estimate].

=== BEFORE ===
- The workstation environment ~/paper1/venvs/paper1-gpu and the data in
  ~/paper1/PhD_data are the same as for the earlier jobs.
- Do NOT run the old JOB_20261119_X-9_e5.zip (it is superseded by DL-46).

=== RUN ===
1. Unzip {JOB_ID}.zip into ~/paper1/job/
2. Open a terminal in ~/paper1/job/{JOB_ID}/
3. Run this ONE command (one line):
     python3.12 run_job.py --base-dir ~/paper1 --data-dir ~/paper1/PhD_data
   Keep the computer on. Do not close the terminal.

What you will see:
- the unit tests ("OK");
- the PREFLIGHT (about 6-8 minutes): 13 short runs, then the line
  "preflight OK";
- the 13 full runs (about 3 minutes each);
- at the end, one line per run ("pass" or "fail") and the line
  "gate: backdoor choice ... (PASS or FAIL); label flip choice ...".
Normal messages you can ignore: "DEPRECATED FEATURE: run_simulation",
"FutureWarning ... RAY_ACCEL_ENV_VAR...", "Manually terminating ClientAppActor".

=== IF IT STOPS ===
- Power cut, or you closed it: run the SAME command again in the SAME folder.
  Finished runs are kept in "state_gate" and are skipped.
- "CUDA out of memory" or a Ray timeout: the job retries each run by itself
  (3 attempts). If a run still fails, it is skipped and the status becomes
  INCOMPLETE: run the same command again.

=== AFTER ===
The job ALWAYS writes {JOB_ID}_results.zip in this folder, also if it fails.
Copy it to the laptop: 05_Experiments/workstation/inbox/
The last line shows the status:
- OK          : the gate passed. Tell Claude; Claude builds the E4 job.
- GATE_FAILED : no attack strength passed. Bring the zip back; you decide
                with Claude (DL-47 (4)). No E4 job is built.
- INCOMPLETE  : run the same command again, then bring the zip back.
- FAILED      : bring the zip back; it contains the error.

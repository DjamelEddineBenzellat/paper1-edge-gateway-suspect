JOB {JOB_ID}
Workstation setup for Paper 1: creates the GPU environment and checks it.
Run it once. You can run it again at any time: it reuses the environment.

=== BEFORE THE FIRST RUN (once) ===
1. NVIDIA driver: open a terminal and type   nvidia-smi
   The top line must show "CUDA Version: 13.2" or higher.
   If it is lower, update the NVIDIA driver first.
2. Python 3.12 (64-bit) must be installed.
   Windows: install it from python.org; check with   py -3.12 --version
   Linux:   check with   python3.12 --version
3. Internet must work (the job downloads about 3-4 GB of packages).
4. At least 15 GB free disk space in the folder you choose below.
5. Choose ONE workstation folder for Paper 1, for example:
   Windows: C:\paper1          Linux: ~/paper1
6. Copy the data ONCE from the laptop (same layout as on the laptop):
   <folder>\PhD_data\ptbxl\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3\   (the whole folder)
   <folder>\PhD_data\medmnist\pneumoniamnist.npz
   The 500 Hz folder "records500" is not needed; you may leave it out.

=== RUN THE JOB ===
1. Unzip {JOB_ID}.zip into <folder>\jobs\
2. Open a terminal in <folder>\jobs\{JOB_ID}\
3. Run ONE command:
   Windows (PowerShell or cmd):
     py -3.12 run_job.py --base-dir C:\paper1 --data-dir C:\paper1\PhD_data
   Linux:
     python3.12 run_job.py --base-dir ~/paper1 --data-dir ~/paper1/PhD_data
   It takes about 15-40 minutes, mostly downloads. Do not close the terminal.

=== BRING BACK ===
When it ends, the last lines show "RESULTS ZIP: ...".
Bring back this file:   <folder>\jobs\{JOB_ID}\{JOB_ID}_results.zip
Put it on the laptop in:  05_Experiments/workstation/inbox/
Bring it back also if the job FAILED: the zip contains the error.

=== AFTER SETUP ===
The environment is in   <folder>\venvs\paper1-gpu\
Later jobs will tell you how to use it.

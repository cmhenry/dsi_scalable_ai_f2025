---
title: "llama_cpp_demo_cluster"
author: "Maël Kubli"
date: "2025-09-24"
---

Runnable demo of llama.cpp inference on the UZH ScienceCluster.
It starts a Singularity-based llama-server and runs a small Python client on a sample dataset.

## Repository Structure
```
llama_cpp_demo_cluster/
├─ README.md
├─ src/
│  ├─ llm_inference.py        # Python client calling llama-server
│  └─ start_job.sh            # SLURM job: container + server + Python + logs
├─ data/
│  └─ sample.csv              # tiny demo dataset
├─ model/
│  └─ example.gguf            # example GGUF (replace with your model)
└─ requirements/
   └─ llama_cpp_cluster.yml   # mamba/conda env spec for cluster
```

### Defaults assumed by src/start_job.sh:
You will have to change the username in the start_job.sh and llm_inference.py files to your own uzh shortname
Project root on cluster: `/scratch/username/llama_cpp_demo_cluster`
Model file: `/scratch/username/llama_cpp_demo_cluster/model/example.gguf`
Logs dir: `/data/username/logs`

## Instructions
### 1) Transfer the repo (FileZilla, SFTP)

**What is FileZilla?**

FileZilla is a free, open-source SFTP client with a simple drag-and-drop GUI. It’s great for moving whole folders (with subfolders) to the cluster without remembering long scp commands.

**Where to get it:**

Download the FileZilla Client (not the Server) from the official site or your OS package manager:
https://filezilla-project.org (Client → macOS/Windows installers).
On Linux: sudo apt install filezilla (Debian/Ubuntu) or use your distro’s package manager.

**Why use FileZilla instead of scp?**

* Easier for folders: drag-and-drop entire directories, preserves structure.
* Robust transfers: pause/resume large uploads; auto-reconnect if your VPN/Wi-Fi blips.
* Visual feedback: progress bars, transfer queue, and clear error messages.
* Fewer mistakes: avoids typos in long remote paths and flags.

**How to connect to the Cluster**

1. Open FileZilla → Site Manager → New Site

2. Protocol: SFTP – SSH File Transfer Protocol

3. Host: cluster.s3it.uzh.ch, Port: 22

4. Logon Type: Ask for password (or Normal)

5. User: your UZH shortname

6. Connect and upload the local folder `llama_cpp_demo_cluster/` to: `/scratch/username/llama_cpp_demo_cluster/`

7. Create the logs directory if needed: `mkdir -p /data/username/logs`

### 2) Access: log in to the ScienceCluster
Once you have a ScienceCluster user account, connect from your terminal using your UZH shortname and AD password:

```
ssh -l shortname cluster.s3it.uzh.ch
```

You will be prompted for your password. No characters are echoed while typing.

### 3) First-time setup (interactive GPU session to build the env)
It’s best to build/solve the environment on a GPU node interactively.

#### 3.1) Load modules and open an interactive GPU shell (1 hour)
```
module av                                  # list available modules (optional)
module load a100                           # GPU constraint module (adjust if different)
module load mamba                          # mamba / conda
# (often also required later for the job) module load cuda
# (and for the container)                 module load singularityce

# Start a 1-hour interactive session on a GPU node:
srun --pty -n 1 -c 4 --time=01:00:00 --gpus=1 --mem=16G bash -l
```
You’re now inside a GPU node shell.

#### 3.2) Create the environment from YAML
```
cd /scratch/username/llama_cpp_demo_cluster
mamba env create -n llama_cpp -f requirements/llama_cpp_cluster.yml
```

Activate and sanity-check:
```
source activate llama_cpp
python -V
python -c "import sys; print('ok', sys.version)"
```

Exit the interactive session when done:
```
exit
```

### 4) Run the demo (batch job via SLURM)
#### 4.1) Load required modules (login node)
```
module av
module load a100
module load mamba
```

#### 4.2) Submit the job
```
sbatch /scratch/mkubli/llama_cpp_demo_cluster/src/start_job.sh
```

What the job does (high level):
* Ensures a Singularity instance (my_llama_instance) is running with --nv and binds your model/ folder.
* Launches llama-server with your GGUF on 127.0.0.1:8080.
* Activates llama_cpp env and runs srun python src/llm_inference.py on the sample dataset.
* Starts a periodic nvidia-smi logger.

#### 4.3) Monitor status and logs
```
# job status
squeue -u shortname

# live logs (adjust job name/ID if you prefer SLURM .out/.err)
tail -f /data/mkubli/logs/llama_hs_main.log
tail -f /data/mkubli/logs/python_llama_hs_script.log
tail -f /data/mkubli/logs/nvidia_smi.log

# container instance state (optional)
singularity instance list
```

### 5) Inputs/outputs
* Model: place your .gguf in model/ and update the path in src/start_job.sh if you rename it.
* Data: data/sample.csv (the Python script expects the demo schema; adapt in llm_inference.py if needed).
* Logs (created by start_job.sh):
    * SLURM: /data/mkubli/logs/%x_%j.out and /data/mkubli/logs/%x_%j.err
    * Main shell log: /data/mkubli/logs/llama_hs_main.log
    * Python log: /data/mkubli/logs/python_llama_hs_script.log
    * GPU telemetry: /data/mkubli/logs/nvidia_smi.log

### 6) Common tweaks
* Different GPU/partition: change module load a100 and the #SBATCH --constraint in start_job.sh as needed.
* Context window / throughput: adjust --ctx-size, --n_gpu_layers, and --threads in the server command.
* Environment name: if you used a different env name than llama_cpp, update source activate ... in start_job.sh.

### 7) Troubleshooting (quick)
* Pending job: not enough GPUs or wrong constraint — check squeue, consider removing or changing --constraint.
* Module names differ: run module av and pick the site-specific names (e.g., cuda/12.x, singularityce/...).
* Conda not found: ensure module load mamba and then source activate llama_cpp.
* Container issues: singularity instance list and inspect /tmp/llama-server.log inside the instance if exposed.
* Exporting an environment can be done via: `mamba env export -n env_name --no-builds > env_name.yml` or on your local machine with `conda env export -n env_name --no-builds > env_name.yml` without any build strings, which is better for compatibility between different operating systems, but this can also be done with the build strings by omitting: `--no-builds`
* Importing an environment can be done with: `mamba env create -f env_name.yml`or `conda env create -f env_name.yml`
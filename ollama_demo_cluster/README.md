---
title: "ollama_demo_cluster"
author: "Maël Kubli"
date: "2025-09-24"
---

Runnable demo of ollama on the UZH Science Cluster.
It starts a Singularity-based ollama-server and runs a small test script to see if the service is up and running. 

## Repository Structure
```
ollama_demo_cluster/
├── README.md
├── data
├── logs
├── models
├── requirements
│   └── ollama_cluster.yml
└── src
    ├── job_ollama_template.sh
    ├── ollama_inference_script_cluster_template.py
    └── start_ollama_script_template.sh
```

## Deaults assumed by src/job_ollama_template.sh
You will have t change the username in the job_ollama_template.sh, atart_ollama_script_template.sh and ollama_inference_script_cluster_tmeplate.py to your own uzh shortname.

## Instructions

### 1) Access: log in to the ScienceCluster
Once you have a ScienceCluster user account, connect from your terminal using your UZH shortname and AD password:

```
ssh -l shortname cluster.s3it.uzh.ch
```

You will be prompted for your password. No characters are echoed while typing.

### 2) First-time setup (interactive GPU session to build the env)
It’s best to build/solve the environment on a GPU node interactively.

#### 2.1) Load modules and open an interactive GPU shell (1 hour)
```
module av                                 # list available modules (optional)
module load gpu                           # GPU constraint module (adjust if different)
module load mamba                         # mamba / conda
module load cuda
# (and for the container)                 module load singularityce

# Start a 1-hour interactive session on a GPU node:
srun --pty -n 1 -c 4 --time=01:00:00 --gpus=1 --mem=16G bash -l
```
You’re now inside a GPU node shell.

#### 2.2) Create the environment from YAML
```
cd /scratch/username/llama_cpp_demo_cluster
mamba env create -n llama_cpp -f requirements/ollama_cluster.yml
```

Activate and sanity-check:
```
source activate ollama_env
python -V
python -c "import sys; print('ok', sys.version)"
```

Exit the interactive session when done:
```
exit
```

### 3) Run the demo (batch job via SLURM)
#### 3.1) Load required modules (login node)
```
module av
module load a100        #Change to GPU different one if A100 is to big or small
module load mamba
```

#### 4.2) Submit the job
```
sbatch /scratch/mkubli/ollama_demo_cluster/src/job_ollama_template.sh
```

What the job does (high level):
* Ensures a Singularity instance is runningand binds your model/ folder which is set to data/username/ollama_env/models.
* Launches ollama-server on 127.0.0.1:11434.
* Activates ollama_env env and runs srun python src/ollama_inference_script_cluster_template.py with a sample instruction.



#!/bin/bash
#SBATCH --job-name=llama_cpp_inference_sample
#SBATCH --gpus=1 --constraint=a100
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/data/mkubli/logs/%x_%j.out  # SLURM log
#SBATCH --error=/data/mkubli/logs/%x_%j.err   # Capture stderr separately

##############################
# Redirect all output (stdout and stderr) to a log file
##############################
exec > >(tee -a /data/mkubli/logs/llama_hs_main.log) 2>&1

##############################
# Load Modules
##############################
module load mamba
module load cuda 
module load singularityce

##############################
# Install LLAMA.CPP Container
##############################
cd /data/mkubli

# Pull the latest version of the container only needed once!
echo "Building llama.sif from ghcr.io/ggerganov/llama.cpp:full-cuda ..."
singularity build llama.sif docker://ghcr.io/ggerganov/llama.cpp:full-cuda

##############################
# Run Container in server mode  
##############################
# Start the container as an instance named my_llama_instance
singularity instance start --nv \
    --bind /scratch/mkubli/llama_cpp_demo_cluster/model:/scratch/mkubli/llama_cpp_demo_cluster/model \
    llama.sif my_llama_instance

# Start the LLAMA server in the background and wait a few seconds to ensure it's up
sleep 5

# Run the llama-server command inside the running instance
singularity exec instance://my_llama_instance bash -c "
    export LD_LIBRARY_PATH=/app:\$LD_LIBRARY_PATH &&
    /app/llama-server \
        --model /scratch/mkubli/llama_cpp_demo_cluster/model/llama3_8b.gguf \
        --host 127.0.0.1 \
        --port 8080 \
        --device CUDA0 \
        --threads 6 \
        --n_gpu_layers 128 \
        --ctx-size 4096 \
        --temp 0.0 \
        --top-p 0.9 \
        --n-predict 256 > /tmp/llama-server.log 2>&1 &"

# For the singularity exec command there are a few other options one can set to see what devices are available:
# --list-devices
# To open the shell of the container use this here: or singularity shell instance://my_llama_instance if you are running an interactive session

# Check if the container is running
sleep 1
echo "Show if the container is running..."
singularity instance list 
##############################
# Wait a few seconds for the server to start
##############################
echo "Waiting 30 seconds for the server to spin up..."
sleep 30

##############################
# Start NVIDIA-SMI Logging in Parallel
##############################

nvidia_smi_log="/data/mkubli/logs/nvidia_smi.log"

echo "Starting NVIDIA-SMI logging every 5 minutes..."
(
    while true; do
        echo "------------------------" >> "$nvidia_smi_log"
        echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')" >> "$nvidia_smi_log"
        nvidia-smi >> "$nvidia_smi_log"
        sleep 300
    done
) &
smi_pid=$!

echo "NVIDIA-SMI logging started with PID: $smi_pid"

##############################
# Activate your local conda env and run Python
##############################
source activate llama_cpp

# Check if the correct GPU is visible
nvcc --version
nvidia-smi

# Run the Python script inside the SLURM-allocated job
cd /scratch/mkubli/llama_cpp_demo_cluster/src || { echo "Directory not found! Exiting."; singularity instance stop my_llama_instance; kill $smi_pid; exit 1; }
echo "Starting Python script at $(date)"
srun python llm_inference.py &>> /data/mkubli/logs/python_llama_hs_script.log
exit_status=$?
echo "Python script completed at $(date) with exit code: $exit_status"

##############################
# Stop NVIDIA-SMI Logging When Everything is Done
##############################
echo "Stopping NVIDIA-SMI logging (PID: $smi_pid)..."
kill $smi_pid
wait $smi_pid 2>/dev/null || true
echo "NVIDIA-SMI logging stopped."

echo "Stopping Singularity instance..."
singularity instance stop my_llama_instance || true

echo "Process exited with status: $exit_status"
exit $exit_status
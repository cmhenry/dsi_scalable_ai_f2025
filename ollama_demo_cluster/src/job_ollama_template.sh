#!/bin/bash
#SBATCH --job-name=ollama_cluster_template
#SBATCH --gpus=1 
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/data/mkubli/logs/%x_%j.out  # SLURM log
#SBATCH --error=/data/mkubli/logs/%x_%j.err   # Capture stderr separately

##############################
# Load Modules
##############################
module load mamba
module load cuda 
module load singularityce

##############################
# Install OLLAMA Container
##############################
cd /data/mkubli

# Pull the latest version of the container only needed once or if you want to pull a new version
echo "Building ollama"
singularity build --sandbox /data/mkubli/ollama docker://ollama/ollama

##############################
# 3) Paths & Names
##############################

##############################
# Run Container in server mode  
##############################
#singularity exec -u --nv /data/mkubli/ollama ollama serve &
#singularity exec -u --nv -e ollama=/data/mkubli/ollama_env /data/mkubli/ollama ollama serve &
singularity exec \
  -u \
  --nv \
  -B /data/mkubli/ollama_env:/data/mkubli/ollama_env \
  --env OLLAMA_MODELS=/data/mkubli/ollama_env/models \
  /data/mkubli/ollama \
  ollama serve &

sleep 10
# Check if the server is running
curl -v http://127.0.0.1:11434/   # should return “Ollama is running”

##############################
# Wait a few seconds for the server to start
##############################
echo "Waiting 30 seconds for the server to spin up..."
sleep 30

##############################
# Start NVIDIA-SMI Logging in Parallel
##############################

nvidia_smi_log="/data/mkubli/logs/nvidia_smi_a.log"

echo "Starting NVIDIA-SMI logging every 180 seconds..."
(
    while true; do
        echo "------------------------" >> "$nvidia_smi_log"
        echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')" >> "$nvidia_smi_log"
        nvidia-smi >> "$nvidia_smi_log"
        sleep 180
    done
) &
smi_pid=$!

echo "NVIDIA-SMI logging started with PID: $smi_pid"

##############################
# Activate your local conda env and run Python
##############################
source activate ollama_env

#pip install --no-cache-dir ollama requests pandas langchain

nvcc --version
nvidia-smi

# Navigate to working directory
cd /scratch/mkubli/ollama_demo_cluster/src || { echo "Directory not found! Exiting."; exit 1; }

# Start the separate job script in the background and log output
./start_ollama_script_template.sh &> /data/mkubli/logs/ollama_main.log &
process_id=$!

# Print the Process ID (PID)
echo "Started start_ollama_script.sh with PID: $process_id"

# Wait for the process to finish and capture exit status
wait $process_id
exit_status=$?

##############################
# Stop NVIDIA-SMI Logging When Everything is Done
##############################
echo "Stopping NVIDIA-SMI logging (PID: $smi_pid)..."
kill $smi_pid
wait $smi_pid 2>/dev/null

echo "NVIDIA-SMI logging stopped."
exit $exit_status

# Log exit status
echo "Process $process_id exited with status: $exit_status"

# Optional: Check if the process is still running
if ps -p $process_id > /dev/null; then
    echo "Process is still running!"
else
    echo "Process completed successfully!"
fi

exit $exit_status
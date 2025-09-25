#!/bin/bash

cd /scratch/mkubli/ollama_demo_cluster/src || { echo "Directory not found! Exiting."; exit 1; }

echo "Starting Python script at $(date)" >> /data/mkubli/logs/python_ollama_logger.log

# Run the Python script inside the SLURM-allocated job
srun python ollama_inference_script_cluster_template.py &>> /data/mkubli/logs/python_ollama_logger.log

echo "Python script completed at $(date)" >> /data/mkubli/logs/python_ollama_logger.log

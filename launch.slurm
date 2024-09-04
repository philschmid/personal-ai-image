#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH --partition=hopper-prod  # Adjust this for your cluster
#SBATCH --output=/fsx/philipp/logs/%x-%j.out # Adjust this for your cluster
#SBATCH --err=/fsx/philipp/logs/%x-%j.err    # Adjust this for your cluster

set -x -e

source ~/.bashrc
micromamba activate ai-toolkit
echo "START TIME: $(date)"

export CMD="python run_inference.py --lora-id philschmid/flux-test-1 --num-images-per-prompt 15"

# force crashing on nccl issues like hanging broadcast
export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=COLL
# export NCCL_SOCKET_NTHREADS=1
# export NCCL_NSOCKS_PERTHREAD=1
# export CUDA_LAUNCH_BLOCKING=1

# Specific configuration optimized for the Hugging Face Compute Cluster
# Be ye warned this may not work on other clusters!
module load cuda/12.1

# srun error handling:
# --wait=60: wait 60 sec after the first task terminates before terminating all remaining tasks
# --kill-on-bad-exit=1: terminate a step if any task exits with a non-zero exit code
SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "

clear; srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "$CMD" 2>&1

echo "END TIME: $(date)"
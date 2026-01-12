#!/bin/bash
#SBATCH --account=bcqp-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --mem=16g
#SBATCH --time=00:05:00
#SBATCH --job-name=best-lstm-1hbv-params
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=1
#SBATCH --output=logs-best/%x-%j.out  # Save output log

module load python/3.11.6
module load cuda/11.8.0
source ~/Differentiable-Hydrological-Model-DRB/diffhydro-env/bin/activate

# Run training and save output to log only
python 07camels_best_lstm_extract_params.py

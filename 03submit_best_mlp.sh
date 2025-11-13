#!/bin/bash
#SBATCH --account=bcqp-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --mem=64g
#SBATCH --time=18:10:00
#SBATCH --job-name=best-mlp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --output=logs-best/%x-%j.out  # Save output log

module load python/3.11.6
module load cuda/11.8.0
source ~/Differentiable-Hydrological-Model-DRB/diffhydro-env/bin/activate

# Run training and save output to log only
python 03camels_best_mlp_hbv.py

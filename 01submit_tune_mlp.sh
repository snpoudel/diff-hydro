#!/bin/bash
#SBATCH --account=bcqp-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --mem=64g
#SBATCH --time=18:10:00
#SBATCH --job-name=tune-mlp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --array=0-2

module load python/3.11.6
module load cuda/11.8.0

source ~/Differentiable-Hydrological-Model-DRB/diffhydro-env/bin/activate

# Tuning MLP nodes
HIDDEN_DIMS_LIST=(1024 2048 4096)
HIDDEN_DIM=${HIDDEN_DIMS_LIST[${SLURM_ARRAY_TASK_ID}]}

LOGFILE=logs/tune_mlp_hbv_hidden_${HIDDEN_DIM}_%j.log
echo "Hidden dim: ${HIDDEN_DIM}" > ${LOGFILE}

# Run training and save output to log only
python 01camels_tune_mlp_hbv.py --hidden_dim ${HIDDEN_DIM} --num_hbv_units 1 >> ${LOGFILE} 2>&1

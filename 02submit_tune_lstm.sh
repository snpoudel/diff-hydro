#!/bin/bash
#SBATCH --account=bcqp-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --mem=64g
#SBATCH --time=18:10:00
#SBATCH --job-name=tune-lstm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --array=0-2

module load python/3.11.6
module load cuda/11.8.0

source ~/Differentiable-Hydrological-Model-DRB/diffhydro-env/bin/activate

# Tuning LSTM hidden size
HIDDEN_DIMS_LIST=(128 256 384)
HIDDEN_DIM=${HIDDEN_DIMS_LIST[${SLURM_ARRAY_TASK_ID}]}

LOGFILE=logs/tune_lstm_hbv_hidden_${HIDDEN_DIM}_%j.log
echo "Hidden dim: ${HIDDEN_DIM}" > ${LOGFILE}

# Run training and save output to log only
python 02camels_tune_lstm_hbv.py --hidden_dim ${HIDDEN_DIM} --num_hbv_units 1 >> ${LOGFILE} 2>&1

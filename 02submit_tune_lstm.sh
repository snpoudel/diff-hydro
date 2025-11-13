#!/bin/bash
#SBATCH --account=bcqp-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --mem=64g
#SBATCH --time=16:10:00
#SBATCH --job-name=tune-lstm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --array=0

module load python/3.11.6
module load cuda/11.8.0

source ~/Differentiable-Hydrological-Model-DRB/diffhydro-env/bin/activate

# Tuning LSTM hidden size
HIDDEN_DIMS_LIST=(512)
HIDDEN_DIM=${HIDDEN_DIMS_LIST[${SLURM_ARRAY_TASK_ID}]}

export HIDDEN_DIM
export NUM_HBV_UNITS=1

LOGFILE=logs-tune/tune_lstm_${NUM_HBV_UNITS}hbv_hidden_${HIDDEN_DIM}_%j.log
echo "Hidden dim: ${HIDDEN_DIM}" > ${LOGFILE}

# Run training and save output to log only
python 02camels_tune_lstm_hbv.py >> ${LOGFILE} 2>&1

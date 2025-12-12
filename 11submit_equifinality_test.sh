#!/bin/bash
#SBATCH --job-name=cams-equifinality_test            # Job name
#SBATCH --output=equifinality_test%j.log       # Output file name (%j expands to jobID)
#SBATCH --error=equifinality_test%j.log        # Error file name (%j expands to jobID)
#SBATCH --time=500:00:00                 # Time limit (HH:MM:SS)
#SBATCH --nodes=1                       #3 Number of nodes
#SBATCH --ntasks=1                   #7 Number of tasks (one for each job), if you don't know numner of tasks beforehand there are ways to make this input dynamic as well
#SBATCH --cpus-per-task=4               #3 Number of CPU cores per task
#SBATCH --mem=0                        # Memory per CPU core (adjust as needed)

# Load necessary modules
# All modules are loaded inside the virtual environment so don't need to load here (check: pip list modules when virtual environment is loaded) 
module load python/3.11.5
# Activate your virtual environment if needed
source ~/pyenv-pytorch/bin/activate

# Run your Python script with mpi
python3 11params_equifinality_test.py
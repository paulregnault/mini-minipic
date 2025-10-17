#!/bin/bash
#SBATCH --job-name=minipic
#SBATCH --account=pqi@a100                   
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --ntasks=1                            
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --time=00:20:00
#SBATCH --output=output
#SBATCH --error=error
#SBATCH --constraint=a100

# ________________________________________________________________________
# Minipic environment

module purge
module load arch/a100
module load nvidia-compilers/24.3
module load cmake/3.31.4
module load python

# ________________________________________________________________________
# Directory where the script was launched

cd "$SLURM_SUBMIT_DIR"

# ________________________________________________________________________
# Launch the test

python run.py -g gpu-a100

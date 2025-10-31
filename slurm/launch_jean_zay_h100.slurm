#!/bin/bash
#SBATCH --job-name=minipic
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --threads-per-core=1
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --time=00:20:00
#SBATCH --output=output
#SBATCH --error=error
#SBATCH --constraint=h100

set -e

# Modules

module purge
module load arch/h100
module load nvidia-compilers/24.7
module load cmake/3.31.4
module load python

# Python env

source ./venv/bin/activate

# Launch the test

mini-run -g gpu-h100

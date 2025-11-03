#!/bin/bash
#SBATCH --job-name=minipic
#SBATCH --output=output
#SBATCH --error=error
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --constraint=MI300
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1

set -e

# Modules

module purge
module load "cpe/24.07"
module load "craype-accel-amd-gfx942"
module load "craype-x86-genoa"
module load "PrgEnv-cray"
module load "amd-mixed"
module load "cmake/3.27.9"
module load "cray-python/3.11.7"

# Python env

if [[ -d ./venv ]]
then
    source ./venv/bin/activate
fi

# Launch the test

mini-run -g gpu-mi300a

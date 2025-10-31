#!/bin/bash
#SBATCH --job-name=minipic
#SBATCH --output=output
#SBATCH --error=error
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --constraint=MI250&turbo
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1

set -e

# Modules

module purge
module load cpe/24.07
module load craype-accel-amd-gfx90a craype-x86-genoa
module load PrgEnv-cray
module load amd-mixed
module load cmake
module load python

# Python env

source ./venv/bin/activate

# Launch the test

mini-run -g gpu-mi250

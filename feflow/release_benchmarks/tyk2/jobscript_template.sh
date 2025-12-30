#!/bin/bash

#SBATCH --partition gpu
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --job-name=feflow-bench-tyk2-legacy
#SBATCH --output=stdout/%A_%a_%x_%N.out
#SBATCH --error=stderr/%A_%a_%x_%N.err
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus-per-task=1
#SBATCH --exclude=iscc006

# Print hostname
echo $SLURM_SUBMIT_HOST

# Source bashrc (useful for working conda/mamba)
source ${HOME}/.bashrc
OPENMM_CPU_THREADS=1

# Activate environment
mamba activate feflow-dev

# Report node in use
hostname

# Open eye license activation/env
export OE_LICENSE=${HOME}/.OpenEye/oe_license.txt

# Report CUDA info
env | sort | grep 'CUDA'

# set loglevel debug
LOGLEVEL=DEBUG

# TEMPLATE ENDS HERE -- Add commands after this line

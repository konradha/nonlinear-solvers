#!/bin/bash

# -- py-mpi4py/3.1.4

#SBATCH --job-name=nlse_mpi
#SBATCH --time=00:12:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --output=nlse_mpi_%j.out
#SBATCH --error=nlse_mpi_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

mpirun -n $SLURM_NTASKS python submit_nlse.py

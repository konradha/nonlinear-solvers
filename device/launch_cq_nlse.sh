#!/bin/bash


#SBATCH --job-name=nlse_cq_mpi
#SBATCH --time=00:12:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --output=nlse_cq_mpi_%j.out
#SBATCH --error=nlse_cq_mpi_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

mpirun -n $SLURM_NTASKS python submit_nlse_cq.py

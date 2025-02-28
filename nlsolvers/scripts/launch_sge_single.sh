#!/bin/bash

#SBATCH --job-name=sg_solver
#SBATCH --time=00:05:00
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --output=sg_solver_%j.out
#SBATCH --error=sg_solver_%j.err
#SBATCH --ntasks=1
#SBATCH --partition=gpu

module load stack/2024-06 cuda/12.1.1 python/3.11.6 eigen
module load python_cuda/3.11.6
module load gcc/12.2.0 
module load cmake/3.27.7

mkdir ${SCRATCH}/sge_dev_${SLURM_JOB_ID}
python sge_single.py --exe=../build/bin/sg_single_dev\
	--nx 256 \
    	--ny 256 \
    	--Lx 10.0 \
    	--Ly 10.0 \
    	--T 10.0 \
    	--nt 1000 \
    	--snapshots 100 \
    	--num-runs 10 \
    	--output-dir ${SCRATCH}/sge_dev_${SLURM_JOB_ID}

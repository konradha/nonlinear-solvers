#!/bin/bash
#SBATCH --job-name=find_nan_kge_2d
#SBATCH --mem-per-cpu=72G
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/nan_%j.out
#SBATCH --error=logs/nan_%j.err

module load stack/2024-06 cuda/12.1.1 python/3.11.6 eigen
module load gcc/12.2.0 openmpi/4.1.6
module load py-mpi4py/3.1.4
module load cmake/3.27.7

python find_nans.py $SCRATCH/kge_2d/ to_prune.txt

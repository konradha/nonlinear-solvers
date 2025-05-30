#!/bin/bash

#SBATCH --job-name=phi4_solver_dev
#SBATCH --time=00:10:00
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --output=logs/phi4_solver_dev_%j.out
#SBATCH --error=logs/phi4_solver_dev_%j.err
#SBATCH --ntasks=1
#SBATCH --partition=gpu

# This script needs to be launched from the nlsolvers/scripts dir.
# The directory tree is hardcoded to ease launches.
mkdir ${SCRATCH}/phi4_dev_${SLURM_JOB_ID}
mkdir -p ../data
mkdir -p ../data/phi4_dev
mkdir -p ../data/phi4_dev/animations

module load stack/2024-06 cuda/12.1.1 python/3.11.6 eigen
module load python_cuda/3.11.6
module load gcc/12.2.0 
module load ffmpeg

python phi4_simplified_launch.py --exe=../build/bin/phi4_dev\
		--nx 256 \
		--ny 256 \
		--Lx 10.0 \
		--Ly 10.0 \
		--T 10.0 \
		--nt 1000 \
		--dr-x 128 --dr-y 128 \
		--m_type "one"\
		--snapshots 25 \
		--num-runs 10 \
		--output-dir ${SCRATCH}/phi4_dev_${SLURM_JOB_ID}

cp ${SCRATCH}/phi4_dev_${SLURM_JOB_ID}/hdf5/* ../data/phi4_dev/
cp ${SCRATCH}/phi4_dev_${SLURM_JOB_ID}/trajectories/* ../data/phi4_dev/animations/


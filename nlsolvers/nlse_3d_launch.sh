#!/bin/bash

#SBATCH --job-name=nlse_3d_dev
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --output=logs/nlse3d_solver_dev_%j.out
#SBATCH --error=logs/nlse3d_solver_dev_%j.err
#SBATCH --ntasks=1
#SBATCH --partition=gpu

module load stack/2024-06 cuda/12.1.1 python/3.11.6 eigen
module load python_cuda/3.11.6
module load gcc/12.2.0 
module load ffmpeg

for p in multi_soliton_state skyrmion_tube; do
	for cp in waveguide anisotropic optimal; do
		python scripts_nlse/launcher_3d.py \
			--exe build/bin/nlse_3d_dev\
		       	--nx 64 --ny 64 --nz 64\
		       	--dr-x 64 --dr-y 64 --dr-z 64\
		       	--Lx 3. --Ly 3. --Lz 3.\
		       	--T 1. --nt 500 --snapshots 50\
		       	--phenomenon ${p}\
		       	--num-runs 8 --visualize --c-m-pair ${cp}
		done
	done

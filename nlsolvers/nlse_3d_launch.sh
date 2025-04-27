#!/bin/bash

#SBATCH --job-name=nlse_3d_dev
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=32G
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
	python scripts_nlse/launcher_3d.py \
			--exe build/bin/nlse_3d_dev\
		       	--nx 200 --ny 200 --nz 200\
		       	--dr-x 100 --dr-y 100 --dr-z 100\
		       	--Lx 3. --Ly 3. --Lz 3.\
		       	--T 1. --nt 500 --snapshots 50\
			--visualize \
		       	--num-runs 4 \
			--output-dir /cluster/scratch/konradha/${p}_constant \
			--seed $((SLURM_JOB_ID + SLURM_ARRAY_TASK_ID))
	done
	#for cp in optimal; do
	#	python scripts_nlse/launcher_3d.py \
	#		--exe build/bin/nlse_3d_dev\
	#	       	--nx 200 --ny 200 --nz 200\
	#	       	--dr-x 100 --dr-y 100 --dr-z 100\
	#	       	--Lx 3. --Ly 3. --Lz 3.\
	#	       	--T 1. --nt 500 --snapshots 50\
	#		--visualize \
	#	       	--phenomenon ${p}\
	#	       	--num-runs 4 --c-m-pair ${cp}\
	#		--output-dir /cluster/scratch/konradha/${p}_${cp}
	#	done
	#done

#!/bin/bash

#SBATCH --job-name=nlse_3d_dev
#SBATCH --time=01:45:00
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

# SIZE consideration:
# n=300 -> (6 * (2 * 8) * 300³) B for SpMV ....
# (> 2GB) which is not even taking the various work data that
# need to be allocated into account ...

n=85
nt=1000
dr=64
exe=build/bin/nlse_3d_dev

# non-constant: i u_t + \Delta u + |u|²u = 0
for p in multi_soliton_state skyrmion_tube; do
        python scripts_nlse/launcher_3d.py \
                --exe ${exe} \
                --nx $n --ny $n --nz $n \
                --dr-x ${dr} --dr-y ${dr} --dr-z ${dr} \
                --Lx 3. --Ly 3. --Lz 3. \
                --T 1. --nt ${nt} --snapshots 100 \
                --num-runs 8 \
		--phenomenon ${p} \
		--visualize --delete-intermediates \
                --seed $((SLURM_JOB_ID + SLURM_ARRAY_TASK_ID)) \
                --output-dir /cluster/scratch/konradha/nlse_3d_test
done

# for p in multi_soliton_state skyrmion_tube; do
#         python scripts_nlse/launcher_3d.py \
#                 --exe build/bin/nlse_3d_dev \
# 		--nx $n --ny $n --nz $n \
#                 --dr-x ${dr} --dr-y ${dr} --dr-z ${dr} \
#                 --Lx 3. --Ly 3. --Lz 3. \
#                 --T 1. --nt ${nt} --snapshots 40 \
#                 --visualize \
#                 --num-runs 2 \
# 		--phenomenon ${p} \
#                 --seed $((SLURM_JOB_ID + SLURM_ARRAY_TASK_ID)) \
#                 --output-dir /cluster/scratch/konradha/nlse_${p}_constant_with_visuals
# done
# 
# 
# # non-constant: i u_t + div(c(x,y,z)grad(u)) + m(x,y,z) |u|²u = 0
# for p in multi_soliton_state skyrmion_tube; do
# 	for cp in optimal sharp_interfaces waveguide grf_threshold anisotropic; do
# 		python scripts_nlse/launcher_3d.py \
# 			--exe build/bin/nlse_3d_dev \
# 			--nx $n --ny $n --nz $n \
# 			--dr-x ${dr} --dr-y ${dr} --dr-z ${dr} \
# 			--Lx 3. --Ly 3. --Lz 3. \
# 			--T 1. --nt ${nt} --snapshots 40 \
# 			--num-runs 4 \
# 			--phenomenon ${p} \
# 			--c-m-pair ${cp} \
# 			--seed $((SLURM_JOB_ID + SLURM_ARRAY_TASK_ID)) \
# 			--output-dir /cluster/scratch/konradha/nlse_${p}_${cp}
# 	done
# done
# 
# for p in multi_soliton_state skyrmion_tube; do
# 	for cp in optimal sharp_interfaces waveguide grf_threshold anisotropic; do
# 		python scripts_nlse/launcher_3d.py \
# 			--exe build/bin/nlse_3d_dev \
# 			--nx $n --ny $n --nz $n \
# 			--dr-x ${dr} --dr-y ${dr} --dr-z ${dr} \
# 			--Lx 3. --Ly 3. --Lz 3. \
# 			--T 1. --nt ${nt} --snapshots 40 \
# 			--visualize \
# 			--num-runs 2 \
# 			--phenomenon ${p} \
# 			--c-m-pair ${cp} \
# 			--seed $((SLURM_JOB_ID + SLURM_ARRAY_TASK_ID)) \
# 			--output-dir /cluster/scratch/konradha/nlse_${p}_${cp}_with_visuals
# 	done
# done

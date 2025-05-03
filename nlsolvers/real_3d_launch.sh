#!/bin/bash

#SBATCH --job-name=kge_3d_dev
#SBATCH --time=00:45:00
#SBATCH --mem-per-cpu=32G
#SBATCH --gpus=1
#SBATCH --output=logs/kge3d_solver_dev_%j.out
#SBATCH --error=logs/kge3d_solver_dev_%j.err
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

n=200
nt=6000 # using SV we're really _fast_
dr=100
exe=build/bin/sp4_sv_3d_dev
L=3.

# constant: u_tt = \Delta u - u
for p in kink_field; do
        python scripts_sge_kge/real_3d_launcher.py \
                --exe ${exe} \
                --nx $n --ny $n --nz $n \
                --dr-x ${dr} --dr-y ${dr} --dr-z ${dr} \
                --Lx $L --Ly $L --Lz $L \
                --T 8. --nt ${nt} --snapshots 40 \
                --num-runs 2 \
		--visualize
		--phenomenon ${p} \
                --seed $((SLURM_JOB_ID + SLURM_ARRAY_TASK_ID)) \
                --output-dir /cluster/scratch/konradha/sp4-test
done

# for p in kink_field; do
# 	python scripts_sge_kge/real_3d_launcher.py \
#                 --exe ${exe} \
#                 --nx $n --ny $n --nz $n \
#                 --dr-x ${dr} --dr-y ${dr} --dr-z ${dr} \
#                 --Lx $L --Ly $L --Lz $L \
#                 --T 8. --nt ${nt} --snapshots 40 \
#                 --num-runs 2 \
# 		--phenomenon ${p} \
# 		--visualize \
#                 --seed $((SLURM_JOB_ID + SLURM_ARRAY_TASK_ID)) \
#                 --output-dir /cluster/scratch/konradha/kge_${p}_constant
# done


# non-constant: u_tt = div(c(x,y,z)grad(u)) - m(x,y,z) u
for p in kink_field; do
	for cp in optimal sharp_interfaces waveguide grf_threshold anisotropic; do
		python scripts_sge_kge/real_3d_launcher.py \
			--exe ${exe} \
			--nx $n --ny $n --nz $n \
			--dr-x ${dr} --dr-y ${dr} --dr-z ${dr} \
			--Lx $L --Ly $L --Lz $L \
			--T 8. --nt ${nt} --snapshots 40 \
			--num-runs 8 \
			--phenomenon ${p} \
			--c-m-pair ${cp} \
			--seed $((SLURM_JOB_ID + SLURM_ARRAY_TASK_ID)) \
			--output-dir /cluster/scratch/konradha/kge_${p}_${cp}
	done
done

# for p in kink_field; do
# 	for cp in optimal sharp_interfaces waveguide grf_threshold anisotropic; do
# 		python scripts_sge_kge/real_3d_launcher.py \
# 			--exe ${exe} \
# 			--nx $n --ny $n --nz $n \
# 			--dr-x ${dr} --dr-y ${dr} --dr-z ${dr} \
# 			--Lx $L --Ly $L --Lz $L \
# 			--T 8. --nt ${nt} --snapshots 40 \
# 			--num-runs 2 \
# 			--visualize \
# 			--phenomenon ${p} \
# 			--c-m-pair ${cp} \
# 			--seed $((SLURM_JOB_ID + SLURM_ARRAY_TASK_ID)) \
# 			--output-dir /cluster/scratch/konradha/kge_${p}_${cp}
# 	done
# done

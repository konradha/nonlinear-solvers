#!/bin/bash

#SBATCH --job-name=integrator_comparison
#SBATCH --time=01:59:00
#SBATCH --mem-per-cpu=32G
#SBATCH --mail-type=END
#SBATCH --mail-user=konradha@yahoo.com
#SBATCH --gpus=1
#SBATCH --output=logs/integrator_comparison_%j.out
#SBATCH --error=logs/integrator_comparison_%j.err
#SBATCH --ntasks=3
#SBATCH --partition=gpu

# more than one task to not have the cpu-only integrator stall
# due to Python subprocess overhead

module load stack/2024-06 cuda/12.1.1 python/3.11.6 eigen
module load python_cuda/3.11.6
module load gcc/12.2.0 
module load ffmpeg


python scripts_nlse/compare_integrators_nlse_3d.py \
       	--exe1 build/bin/nlse_3d_dev \
	--exe2 build/bin/nlse_cubic_host_3d \
       	--name1 SS2-dev --name2 SS2-host\
       	--nx 64 --ny 64 --nz 64\
       	--Lx .1 --Ly .1 --Lz .1\
       	--T .1 --nt 10 --num-snapshots 10 \
	--output-dir ${SCRATCH}/compare_ss2_host_dev_nlse_${SLURM_JOB_ID}

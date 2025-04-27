#!/bin/bash

#SBATCH --job-name=integrator_comparison
#SBATCH --time=02:59:00
#SBATCH --mem-per-cpu=32G
#SBATCH --mail-type=END
#SBATCH --mail-user=konradha@yahoo.com

#SBATCH --output=logs/integrator_comparison_%j.out
#SBATCH --error=logs/integrator_comparison_%j.err
#SBATCH --ntasks=4
#SBATCH --partition=gpu

module load stack/2024-06 cuda/12.1.1 python/3.11.6 eigen
module load python_cuda/3.11.6
module load gcc/12.2.0 
module load ffmpeg


python scripts_nlse/compare_integrators_nlse_3d.py \
       	--exe1 build/bin/nlse_cubic_host_3d \
	--exe2 build/bin/nlse_cubic_host_gautschi_3d\
       	--name1 SS2-host --name2 sEWI-host\
       	--nx 150 --ny 150 --nz 150\
       	--Lx 3. --Ly 3. --Lz 3.\
       	--T .5 --nt 500 \
	--output-dir ${SCRATCH}/compare_ss2_sewi_host_nlse_${SLURM_JOB_ID}

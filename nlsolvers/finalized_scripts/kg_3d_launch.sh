#!/bin/bash

#SBATCH --job-name=kge_3d_dev
#SBATCH --time=03:30:00
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=1
#SBATCH --output=logs/%j/kge3d_solver_dev.out
#SBATCH --error=logs/%j/kge3d_solver_dev.err
#SBATCH --partition=gpu

BASE_DIR="$1"

if [ -z "$BASE_DIR" ]; then
    echo "sbatch kg_3d_launch.sh exe-dir"
    exit 1
fi


module load stack/2024-06 cuda/12.1.1 python/3.11.6 eigen
module load python_cuda/3.11.6
module load gcc/12.2.0 
module load ffmpeg

L=3.

n=220
nt=10000

dr=128
exe=${BASE_DIR}

T=10.
snap=40

# this should yield about 0.5 TB
# and run in about 2hrs

for c_type in constant piecewise_constant layered waveguide quasiperiodic turbulent; do
	for m_type in constant piecewise gradient topological defects quasiperiodic multiscale; do
		time {		
			for p in kink_field; do
				for v in random; do
			        	python real_launcher_3d.py \
			        	        --system-type klein_gordon --phenomenon ${p} --velocity-type ${v} \
						--anisotropy-type ${c_type} --m_type ${m_type} \
			        	        --nx $n --ny $n --nz $n --Lx $L --Ly $L --Lz $L \
					       	--nt ${nt} --T ${T} --snapshots ${snap} \
			        	        --dr-x ${dr} --dr-y ${dr} --dr-z ${dr} --delete-intermediates \
			        	        --exe ${exe} --num-runs 1 \
			        	        --output-dir=$SCRATCH/kge_3d/c_${c_type}/m_${m_type}/${p}
				done
			done
		}
	done
done

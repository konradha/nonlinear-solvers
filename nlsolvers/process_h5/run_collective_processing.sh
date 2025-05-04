#!/bin/bash
#SBATCH --job-name=energy_stats
#SBATCH --ntasks=12
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=32G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%j/energy_stats.log
#SBATCH --error=logs/%j/energy_stats.err

module load stack/2024-06 cuda/12.1.1 python/3.11.6 eigen
module load gcc/12.2.0  openmpi/4.1.6
module load py-mpi4py/3.1.4
module load cmake/3.27.7

BASE_DIR="$1"
OUTPUT_DIR="$2"

if [ -z "$BASE_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "sbatch submit_energy_stats.sh <hdf5_directory> <output_directory>"
    exit 1
fi

# for about 150HDF5 files and about 250GB data this nice script ran in less than 2 min! 

mkdir -p "$OUTPUT_DIR"

mpirun -n $SLURM_NTASKS python collective_stats_per_dir.py "$BASE_DIR" "$OUTPUT_DIR"

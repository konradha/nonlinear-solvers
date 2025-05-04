#!/bin/bash
#SBATCH --mem-per-cpu=8G
#SBATCH --time=00:10:00

file=$1
output_dir="$SCRATCH/trajectory_slice_plots/$(basename $(dirname $(dirname $file)))"

python plot_slices.py "$file" "$output_dir"

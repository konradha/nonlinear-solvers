sbatch --array=1-$(wc -l < h5_file_list.txt) --mem-per-cpu=8G --time=00:10:00 \
  --wrap 'file=$(sed -n ${SLURM_ARRAY_TASK_ID}p h5_file_list.txt); \
  output_dir="$SCRATCH/trajectory_slice_plots/$(basename $(dirname $(dirname $file)))"; \
  python plot_slices.py "$file" "$output_dir"'

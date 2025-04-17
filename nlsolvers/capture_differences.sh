#!/bin/bash
set -e 
trap 'kill 0' EXIT

for outer_i in $(seq 1 2); do 
  #for p in multi_soliton spectral chaotic vortex_lattice ring_soliton multi_ring dark_soliton solitary_wave_with_ambient free_singularity_adapted logarithmic_singularity_adapted turbulent_condensate topological_defect_network akhmediev_breather self_similar_pattern; do 
  #for p in spectral; do
  for p in spectral; do
    for inner_i in 1 1; do 
      python scripts_nlse/compare_integrators_nlse.py\
          --exe1 build/bin/nlse_cubic --name1 SS2\
          --exe2 build/bin/nlse_cubic_gautschi --name2 sEWI\
          --Lx 5. --Ly 5. --nx 100 --ny 100 --T 1. --nt 1000 --num-snapshots 100\
          --ic-type $p\
          --output-dir ${p}_diff_special_test
    done
  done &
done

wait || exit 1

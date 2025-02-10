#!/bin/bash

#SBATCH --job-name=complex_lanczos_test  
#SBATCH --time=00:12:00         
#SBATCH --mem-per-cpu=4G                          
#SBATCH --gpus=1    
#SBATCH --output=complex_lanczos_test_%j.out     
#SBATCH --error=complex_lanczos_test_%j.err       


cmd="./to_lanczos_complex_test"
nvprof ${cmd}

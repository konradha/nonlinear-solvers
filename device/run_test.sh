#!/bin/bash

#SBATCH --job-name=nlse_test  
#SBATCH --time=00:12:00         
#SBATCH --mem-per-cpu=4G                          
#SBATCH --gpus=1    
#SBATCH --output=test_nlse_%j.out     
#SBATCH --error=test_nlse_%j.err       


#cmd="./to_lanczos_complex_test"
#nvprof ${cmd}
#compute-sanitizer ./to_matfunc_real_test
./to_nlse_driver

#!/bin/bash

#SBATCH --job-name=nlse_test_par16  
#SBATCH --time=00:12:00         
#SBATCH --mem-per-cpu=4G                          
#SBATCH --gpus=1    
#SBATCH --output=test_nlse_par16_%j.out     
#SBATCH --error=test_nlse_par16_%j.err       
#SBATCH --cpus-per-task=16


#cmd="./to_lanczos_complex_test"
#nvprof ${cmd}
#compute-sanitizer ./to_matfunc_real_test
./to_nlse_driver_par

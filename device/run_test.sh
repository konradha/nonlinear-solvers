#!/bin/bash

#SBATCH --job-name=matfunc_complex_test  
#SBATCH --time=00:12:00         
#SBATCH --mem-per-cpu=4G                          
#SBATCH --gpus=1    
#SBATCH --output=matfunc_complex_%j.out     
#SBATCH --error=matfunc_complex_%j.err       


#cmd="./to_lanczos_complex_test"
#nvprof ${cmd}
#compute-sanitizer ./to_matfunc_real_test
compute-sanitizer ./to_matfunc_complex_test

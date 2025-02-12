#!/bin/bash

#SBATCH --job-name=matfunc_complex_test  
#SBATCH --time=00:12:00         
#SBATCH --mem-per-cpu=4G                          
#SBATCH --gpus=1    
#SBATCH --output=matfunc_complex_test_%j.out     
#SBATCH --error=matfunc_complex_test_%j.err       


#cmd="./to_lanczos_complex_test"
#nvprof ${cmd}
./to_matfunc_complex_test

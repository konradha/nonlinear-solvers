#!/bin/bash

#SBATCH --job-name=matfunc_real_test_complex  
#SBATCH --time=00:15:00         
#SBATCH --mem-per-cpu=4G                          
#SBATCH --gpus=1    
#SBATCH --output=matfunc_real_test_complex_%j.out     
#SBATCH --error=matfunc_real_test_complex_%j.err       

./to_matfunc_real_test

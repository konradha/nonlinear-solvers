#!/bin/bash

#SBATCH --job-name=real_matfunc_test  
#SBATCH --time=00:03:00         
#SBATCH --mem-per-cpu=4G                          
#SBATCH --gpus=1    
#SBATCH --output=real_matfunc_test_%j.out     
#SBATCH --error=real_matfunc_test_%j.err       


./to_test_matfunc_real

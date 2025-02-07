#!/bin/bash

#SBATCH --job-name=test_mf_real  
#SBATCH --time=00:15:00         
#SBATCH --mem-per-cpu=4G                          
#SBATCH --gpus=1    
#SBATCH --output=test_mf_real_%j.out     
#SBATCH --error=test_mf_real_%j.err       

./to_test_matfunc_real

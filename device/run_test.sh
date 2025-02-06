#!/bin/bash

#SBATCH --job-name=lanczos_test_complex  
#SBATCH --time=00:15:00         
#SBATCH --mem-per-cpu=4G                          
#SBATCH --gpus=1    
#SBATCH --output=lanczos_test_complex_%j.out     
#SBATCH --error=lanczos_test_complex_%j.err       

./to_lanczos_complex_test

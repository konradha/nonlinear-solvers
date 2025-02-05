#!/bin/bash

#SBATCH --job-name=lanczos_test  
#SBATCH --time=00:15:00         
#SBATCH --mem-per-cpu=4G                          
#SBATCH --gpus=1    
#SBATCH --output=lanczos_test_%j.out     
#SBATCH --error=lanczos_test_%j.err       

./to_lanczos_test

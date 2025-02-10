#!/bin/bash

#SBATCH --job-name=sg_driver_test  
#SBATCH --time=00:12:00         
#SBATCH --mem-per-cpu=4G                          
#SBATCH --gpus=1    
#SBATCH --output=sg_driver_test_%j.out     
#SBATCH --error=sg_driver_test_%j.err       


./to_sg_driver_dev

#!/bin/bash

#SBATCH --job-name=test_sg_driver_dev  
#SBATCH --time=00:12:00         
#SBATCH --mem-per-cpu=4G                          
#SBATCH --gpus=1    
#SBATCH --output=test_sg_driver_dev_%j.out     
#SBATCH --error=test_sg_driver_dev_%j.err       


./to_sg_driver_dev

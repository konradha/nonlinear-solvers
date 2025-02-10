#!/bin/bash

#SBATCH --job-name=matfunc_tester  
#SBATCH --time=00:12:00         
#SBATCH --mem-per-cpu=4G                          
#SBATCH --gpus=1    
#SBATCH --output=matfunc_tester_%j.out     
#SBATCH --error=matfunc_tester_%j.err       


./to_precise_matfunc_tester

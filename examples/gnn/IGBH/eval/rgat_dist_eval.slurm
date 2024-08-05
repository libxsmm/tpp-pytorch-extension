#!/bin/bash

#SBATCH -e slurm-%j.err
#SBATCH -D <full-path-to-current-dir>
#SBATCH -J gnn_conv
#SBATCH --get-user-env
#SBATCH --partition=emr
#SBATCH --nodes=1

python --version
#Example command line 
#sbatch rgat_dist_eval.slurm <full-path-to-dataset> 2 2 1 ./model_medium.pt 0.025 20,20,20

bash run_dist_eval.sh $1 $2 $3 $4 $5 $6 $7

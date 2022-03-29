#! /bin/bash

#SBATCH --job-name=cvpr18_maxusw_solver
#SBATCH --output=/mnt/nfs/scratch1/jpayan/cvpr18_max_usw_solver.out
#SBATCH --error=/mnt/nfs/scratch1/jpayan/cvpr18_max_usw_solver.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=swarm082
#SBATCH --mem=10G
#SBATCH --time=00-10:00:00
#SBATCH --partition=defq

DATASET=cvpr18
DATA_DIR=/mnt/nfs/scratch1/jpayan/MinimalBidding
QUERY_MODEL=random

./submitSolver.sh $DATASET $DATA_DIR $QUERY_MODEL

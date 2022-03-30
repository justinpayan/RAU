#! /bin/bash

#SBATCH --job-name=cvpr18_solver_superstar
#SBATCH --output=/mnt/nfs/scratch1/jpayan/logs/MinimalBidding/cvpr18_solver_superstar.out
#SBATCH --error=/mnt/nfs/scratch1/jpayan/logs/MinimalBidding/cvpr18_solver_superstar.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=swarm082
#SBATCH --mem=10G
#SBATCH --time=00-10:00:00
#SBATCH --partition=defq

DATASET=cvpr18
DATA_DIR=/mnt/nfs/scratch1/jpayan/MinimalBidding
QUERY_MODEL=superstar

./submitSolver.sh $DATASET $DATA_DIR $QUERY_MODEL

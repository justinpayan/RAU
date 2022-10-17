#! /bin/bash

#SBATCH --job-name=cvpr_minmaxsolver
#SBATCH --output=/mnt/nfs/scratch1/jpayan/logs/MinimalBidding/cvpr_minmaxsolver.out
#SBATCH --error=/mnt/nfs/scratch1/jpayan/logs/MinimalBidding/cvpr_minmaxsolver.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=swarm080
#SBATCH --time=00-06:00:00
#SBATCH --partition=longq

DATASET=cvpr
DATA_DIR=/mnt/nfs/scratch1/jpayan/MinimalBidding

./runMaxMinSolver.sh $DATASET $DATA_DIR
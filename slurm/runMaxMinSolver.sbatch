#! /bin/bash

#SBATCH --job-name=midl_minmaxsolver
#SBATCH --output=/mnt/nfs/scratch1/jpayan/logs/MinimalBidding/midl_minmaxsolver.out
#SBATCH --error=/mnt/nfs/scratch1/jpayan/logs/MinimalBidding/midl_minmaxsolver.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=swarm080
#SBATCH --time=00-06:00:00
#SBATCH --partition=longq

DATASET=midl
DATA_DIR=/mnt/nfs/scratch1/jpayan/MinimalBidding

./runMaxMinSolver.sh $DATASET $DATA_DIR
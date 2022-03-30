#! /bin/bash

#SBATCH --job-name=cvpr_maxexpected
#SBATCH --output=/mnt/nfs/scratch1/jpayan/logs/MinimalBidding/cvpr_maxexpected.out
#SBATCH --error=/mnt/nfs/scratch1/jpayan/logs/MinimalBidding/cvpr_maxexpected.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00-01:00:00
#SBATCH --partition=defq

DATASET=cvpr
DATA_DIR=/mnt/nfs/scratch1/jpayan/MinimalBidding
QUERY_MODEL=supergreedymax

./submitExp.sh $DATASET $DATA_DIR $QUERY_MODEL

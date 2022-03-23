#! /bin/bash

#SBATCH --job-name=cvpr_maxusw_2
#SBATCH --output=/mnt/nfs/scratch1/jpayan/cvpr_max_usw_2.out
#SBATCH --error=/mnt/nfs/scratch1/jpayan/cvpr_max_usw_2.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --ntasks=1
#SBATCH --time=00-01:00:00
#SBATCH --partition=defq

NUM_PROCS=20

DATASET=cvpr
DATA_DIR=/mnt/nfs/scratch1/jpayan/MinimalBidding

./submitExp.sh $DATASET $DATA_DIR ${NUM_PROCS}

#! /bin/bash

#SBATCH --job-name=cvpr_maxusw_2
#SBATCH --output=/mnt/nfs/scratch1/jpayan/cvpr_max_usw_2.out
#SBATCH --error=/mnt/nfs/scratch1/jpayan/cvpr_max_usw_2.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --ntasks=1
#SBATCH --time=06-00:00:00
#SBATCH --partition=longq

NUM_PROCS=10

DATASET=cvpr
DATA_DIR=/mnt/nfs/scratch1/jpayan/MinimalBidding

./submitExp.sh $DATASET $DATA_DIR ${NUM_PROCS}

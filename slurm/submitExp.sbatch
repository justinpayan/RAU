#! /bin/bash

#SBATCH --job-name=cvpr_maxusw
#SBATCH --output=/mnt/nfs/scratch1/jpayan/cvpr_max_usw.out
#SBATCH --error=/mnt/nfs/scratch1/jpayan/cvpr_max_usw.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=06-00:00:00
#SBATCH --partition=longq

NUM_PROCS=20

DATASET=cvpr
DATA_DIR=/mnt/nfs/scratch1/jpayan/MinimalBidding

./submitExp.sh $DATASET $DATA_DIR ${NUM_PROCS}

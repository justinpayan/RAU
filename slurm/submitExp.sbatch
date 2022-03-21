#! /bin/bash

#SBATCH --job-name=cvpr_maxusw
#SBATCH --output=/mnt/nfs/scratch1/jpayan/cvpr_max_usw.out
#SBATCH --error=/mnt/nfs/scratch1/jpayan/cvpr_max_usw.err
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --mem=40G
#SBATCH --time=06-00:00:00
#SBATCH --partition=longq

NUM_PROCS=20

export MKL_NUM_THREADS=${NUM_PROCS}
export OPENBLAS_NUM_THREADS=${NUM_PROCS}
export OMP_NUM_THREADS=${NUM_PROCS}

DATASET=cvpr
DATA_DIR=/mnt/nfs/scratch1/jpayan/MinimalBidding

./submitExp.sh $DATASET $DATA_DIR ${NUM_PROCS}

#! /bin/bash

#SBATCH --job-name=midl_maxusw
#SBATCH --output=/mnt/nfs/scratch1/jpayan/midl_max_usw.out
#SBATCH --error=/mnt/nfs/scratch1/jpayan/midl_max_usw.err
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --mem=10G
#SBATCH --time=00-02:00:00
#SBATCH --partition=defq

NUM_PROCS=20

export MKL_NUM_THREADS=${NUM_PROCS}
export OPENBLAS_NUM_THREADS=${NUM_PROCS}
export OMP_NUM_THREADS=${NUM_PROCS}

DATASET=midl
DATA_DIR=/mnt/nfs/scratch1/jpayan/MinimalBidding

./submitExp.sh $DATASET $DATA_DIR ${NUM_PROCS}

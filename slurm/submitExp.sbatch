#! /bin/bash

#SBATCH --job-name=midl_maxusw
#SBATCH --output=/mnt/nfs/scratch1/jpayan/midl_max_usw.out
#SBATCH --error=/mnt/nfs/scratch1/jpayan/midl_max_usw.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=10G
#SBATCH --time=00-02:00:00
#SBATCH --partition=defq

DATASET=midl
DATA_DIR=/mnt/nfs/scratch1/jpayan/MinimalBidding

./submitExp.sh $DATASET $DATA_DIR

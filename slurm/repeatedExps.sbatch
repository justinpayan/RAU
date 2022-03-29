#! /bin/bash

#SBATCH --job-name=repeated_exps_cvpr_random
#SBATCH --output=/mnt/nfs/scratch1/jpayan/repeated_exps_cvpr_random.out
#SBATCH --error=/mnt/nfs/scratch1/jpayan/repeated_exps_cvpr_random.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00-10:00:00
#SBATCH --partition=defq

DATASET=cvpr
DATA_DIR=/mnt/nfs/scratch1/jpayan/MinimalBidding
QUERY_MODEL=random

./repeatedExps.sh $DATASET $DATA_DIR $QUERY_MODEL

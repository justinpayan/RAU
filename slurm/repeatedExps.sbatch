#! /bin/bash

SEED=$1
DATASET=$2
QUERY_MODEL=$3
DATA_DIR=/mnt/nfs/scratch1/jpayan/MinimalBidding

sbatch --job-name=repeated_exps_${DATASET}_${QUERY_MODEL}_${SEED} \
       --output=/mnt/nfs/scratch1/jpayan/repeated_exps_${DATASET}_${QUERY_MODEL}_${SEED}.out \
       --error=/mnt/nfs/scratch1/jpayan/repeated_exps_${DATASET}_${QUERY_MODEL}_${SEED}.err \
       --nodes=1 \
       --ntasks=1 \
       --time=05-00:00:00 \
       --partition=longq \
       ./repeatedExps.sh $DATASET $DATA_DIR $QUERY_MODEL $SEED

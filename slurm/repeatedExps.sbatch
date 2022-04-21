#! /bin/bash

SEED=$1
DATASET=$2
QUERY_MODEL=$3
DATA_DIR=/mnt/nfs/scratch1/jpayan/MinimalBidding
LOG_DIR=/mnt/nfs/scratch1/jpayan/logs/MinimalBidding

sbatch --job-name=repeated_exps_${DATASET}_${QUERY_MODEL}_${SEED} \
       --output=${LOG_DIR}/repeated_exps_${DATASET}_${QUERY_MODEL}_${SEED}.out \
       --error=${LOG_DIR}/repeated_exps_${DATASET}_${QUERY_MODEL}_${SEED}.err \
       --nodes=1 \
       --ntasks=1 \
       --mem=10G \
       --time=00-10:00:00 \
       --partition=defq \
       ./repeatedExps.sh $DATASET $DATA_DIR $QUERY_MODEL $SEED

#! /bin/bash

SEED=$1
DATASET=$2
DATA_DIR=/mnt/nfs/scratch1/jpayan/MinimalBidding
LOG_DIR=/mnt/nfs/scratch1/jpayan/logs/MinimalBidding

sbatch --job-name=repeated_baselines_${DATASET}_${SEED} \
       --output=${LOG_DIR}/repeated_baselines_${DATASET}_${SEED}.out \
       --error=${LOG_DIR}/repeated_baselines_${DATASET}_${SEED}.err \
       --nodes=1 \
       --ntasks=1 \
       --nodelist=swarm082 \
       --time=00-10:00:00 \
       --mem=30G \
       --partition=defq \
       ./repeatedBaselines.sh $DATASET $DATA_DIR $SEED

#! /bin/bash

SEED=$1
DATASET=$2
DATA_DIR=/mnt/nfs/scratch1/jpayan/MinimalBidding

sbatch --job-name=repeated_baselines_${DATASET}_${SEED} \
       --output=/mnt/nfs/scratch1/jpayan/repeated_baselines_${DATASET}_${SEED}.out \
       --error=/mnt/nfs/scratch1/jpayan/repeated_baselines_${DATASET}_${SEED}.err \
       --nodes=1 \
       --ntasks=1 \
       --nodelist=swarm082 \
       --time=00-10:00:00 \
       --partition=defq \
       ./repeatedExps.sh $DATASET $DATA_DIR $SEED

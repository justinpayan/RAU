#! /bin/bash

DATASET=$1
QUERY_MODEL=$2
DATA_DIR=/mnt/nfs/scratch1/jpayan/MinimalBidding
LOG_DIR=/mnt/nfs/scratch1/jpayan/logs/MinimalBidding

sbatch --job-name=solver_${DATASET}_${QUERY_MODEL}_${SEED} \
       --output=${LOG_DIR}/solver_${DATASET}_${QUERY_MODEL}_${SEED}.out \
       --error=${LOG_DIR}/solver_${DATASET}_${QUERY_MODEL}_${SEED}.err \
       --nodes=1 \
       --ntasks=1 \
       --nodelist=swarm082 \
       --mem=30G \
       --time=00-10:00:00 \
       --partition=defq \
       ./submitSolver.sh $DATASET $DATA_DIR $QUERY_MODEL

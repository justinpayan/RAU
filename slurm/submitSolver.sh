#!/usr/bin/env bash

DATASET=$1
DATA_DIR=$2
QUERY_MODEL=$3

for SEED in {0..9}; do
  python ../main.py --mode final_solver --dset_name $DATASET --data_dir $DATA_DIR --query_model $QUERY_MODEL --seed $SEED
done
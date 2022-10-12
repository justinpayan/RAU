#!/usr/bin/env bash

DATASET=$1
DATA_DIR=$2

for SEED in {0..10}; do
  python ../main.py --dset_name $DATASET --data_dir $DATA_DIR --seed $SEED
done
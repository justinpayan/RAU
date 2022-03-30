#!/usr/bin/env bash

DATASET=$1
DATA_DIR=$2
SEED=$3

python ../main.py --mode basic_baselines --dset_name $DATASET --data_dir $DATA_DIR --seed $SEED

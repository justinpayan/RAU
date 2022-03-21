#!/usr/bin/env bash

DATASET=$1
DATA_DIR=$2
NUM_PROCS=$3

python ../main.py --dset_name $DATASET --data_dir $DATA_DIR --num_procs $NUM_PROCS
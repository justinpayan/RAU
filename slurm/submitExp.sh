#!/usr/bin/env bash

DATASET=$1
DATA_DIR=$2
QUERY_MODEL=$3

python ../main.py --mode query_exps --dset_name $DATASET --data_dir $DATA_DIR --query_model $QUERY_MODEL
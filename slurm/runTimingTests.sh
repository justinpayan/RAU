#!/usr/bin/env bash

DATA_DIR=$1
SEED=$2
YEAR=$3
CACHESTR=$4
DYKSTR=$5

python ../timing_exps.py --data_dir $DATA_DIR --seed $SEED --year $YEAR $CACHESTR $DYKSTR

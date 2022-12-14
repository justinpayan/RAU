#!/usr/bin/env bash

DATA_DIR=$1
SEED=$2
YEAR=$3

python ../main_iclr_tests.py --data_dir $DATA_DIR --seed $SEED --year $YEAR

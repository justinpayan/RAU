#!/usr/bin/env bash

module load gurobi/1001

DATA_DIR=$1
SEED=$2
NUM_DUMMIES_BY_TEN=$3

python ../main_bias_tests.py --data_dir $DATA_DIR --seed $SEED --num_dummy_revs $NUM_DUMMIES_BY_TEN

#!/usr/bin/env bash

module load gurobi/1001

DATA_DIR=$1
SEED=$2
NUM_DUMMIES=$3
CONF=$4

python ../main_bias_tests.py --data_dir $DATA_DIR --seed $SEED --num_dummy_revs $NUM_DUMMIES --conf $CONF

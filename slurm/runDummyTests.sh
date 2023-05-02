#!/usr/bin/env bash

module load gurobi/1001

DATA_DIR=$1
SEED=$2
NUM_DUMMIES=$3
CONF=$4
REVSPAPS=$5

python ../main_bias_tests.py --data_dir $DATA_DIR --seed $SEED --num_dummies $NUM_DUMMIES --conf $CONF --revs_or_paps $REVSPAPS

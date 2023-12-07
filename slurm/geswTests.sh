#!/usr/bin/env bash

DATA_DIR=$1
YEAR=$2
ALGO=$3
R_IDX=$4

module load gurobi/1001

python ../egalitarian_welfare_tests.py --data_dir $DATA_DIR --year $YEAR --algo $ALGO --r_idx $R_IDX

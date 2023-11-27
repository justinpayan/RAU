#!/usr/bin/env bash

DATA_DIR=$1
YEAR=$2

module load gurobi/1001

python ../egalitarian_welfare_tests.py --data_dir $DATA_DIR --year $YEAR

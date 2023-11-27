#!/usr/bin/env bash

module load gurobi/1001

DATA_DIR=$1
SEED=$2
YEAR=$3

python ../comparetimedualregular.py --data_dir $DATA_DIR --seed $SEED --year $YEAR

#!/usr/bin/env bash

for SEED in {0..9}; do
    ./repeatedExps.sbatch $SEED cvpr supergreedymax
    ./repeatedExps.sbatch $SEED cvpr18 supergreedymax
done

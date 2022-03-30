#!/usr/bin/env bash

for SEED in {0..9}; do
    ./repeatedBaselines.sbatch $SEED midl
    ./repeatedBaselines.sbatch $SEED cvpr
    ./repeatedBaselines.sbatch $SEED cvpr18
done

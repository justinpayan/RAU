#!/usr/bin/env bash

MODEL=random

for SEED in {0..9}; do
  ./repeatedExps.sbatch $SEED midl $MODEL
  ./repeatedExps.sbatch $SEED cvpr $MODEL
  ./repeatedExps.sbatch $SEED cvpr18 $MODEL
done

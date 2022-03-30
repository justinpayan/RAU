#!/usr/bin/env bash

for SEED in {0..9}; do
  ./repeatedExps.sbatch $SEED midl random
  ./repeatedExps.sbatch $SEED cvpr random
  ./repeatedExps.sbatch $SEED cvpr18 random
  ./repeatedExps.sbatch $SEED midl tpms
  ./repeatedExps.sbatch $SEED cvpr tpms
  ./repeatedExps.sbatch $SEED cvpr18 tpms
  ./repeatedExps.sbatch $SEED midl superstar
  ./repeatedExps.sbatch $SEED cvpr superstar
  ./repeatedExps.sbatch $SEED cvpr18 superstar
done

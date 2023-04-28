#! /bin/bash

DATA_DIR=/mnt/nfs/scratch1/jpayan/MinimalBidding
LOG_DIR=/mnt/nfs/scratch1/jpayan/logs/MinimalBidding

for SEED in {0..0}; do
  for YEAR in {2018..2022}; do
    sbatch --time=05-12:00:00 --partition=longq \
    --nodes=1 --ntasks=1 --mem=40G --output=$LOG_DIR/iclr_tests_${YEAR}_${SEED}_comparetime.out \
    --error=$LOG_DIR/iclr_tests_${YEAR}_${SEED}_comparetime.err --job-name=iclr_tests_${YEAR}_${SEED} \
    ./runICLRTests.sh $DATA_DIR $SEED $YEAR
  done
done

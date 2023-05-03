#! /bin/bash

DATA_DIR=/mnt/nfs/scratch1/jpayan/MinimalBidding
LOG_DIR=/mnt/nfs/scratch1/jpayan/logs/MinimalBidding

for SEED in {0..99}; do
  for YEAR in {2018..2023}; do
    sbatch --time=05-11:00:00 --partition=longq \
    --nodes=1 --ntasks=1 --mem=30G --output=$LOG_DIR/base_iclr_tests_${YEAR}_${SEED}.out \
    --error=$LOG_DIR/base_iclr_tests_${YEAR}_${SEED}.err --job-name=base_iclr_tests_${YEAR}_${SEED} \
    ./runBaselineTests.sh $DATA_DIR $SEED $YEAR
  done
done
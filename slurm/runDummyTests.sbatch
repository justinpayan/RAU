#! /bin/bash

DATA_DIR=/mnt/nfs/scratch1/jpayan/MinimalBidding
LOG_DIR=/mnt/nfs/scratch1/jpayan/logs/MinimalBidding

for SEED in {0..50}; do
  for NUM_DUMMIES_BY_TEN in {0..20}; do
    sbatch --time=00-11:00:00 --partition=defq \
    --nodes=1 --ntasks=1 --mem=20G --output=$LOG_DIR/dummy_tests_${NUM_DUMMIES_BY_TEN}_${SEED}.out \
    --error=$LOG_DIR/dummy_tests_${NUM_DUMMIES_BY_TEN}_${SEED}.err --job-name=dummy_tests_${NUM_DUMMIES_BY_TEN}_${SEED} \
    ./runDummyTests.sh $DATA_DIR $SEED $NUM_DUMMIES_BY_TEN
  done
done
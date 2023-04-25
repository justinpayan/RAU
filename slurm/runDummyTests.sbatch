#! /bin/bash

DATA_DIR=/mnt/nfs/scratch1/jpayan/MinimalBidding
LOG_DIR=/mnt/nfs/scratch1/jpayan/logs/MinimalBidding

CONF="midl"

for SEED in {0..99}; do
  for NUM_DUMMIES in {0..100..5}; do
    sbatch --time=00-11:00:00 --partition=defq \
    --nodes=1 --ntasks=1 --mem=20G --output=$LOG_DIR/dummy_tests_${CONF}_${NUM_DUMMIES}_${SEED}.out \
    --error=$LOG_DIR/dummy_tests_${CONF}_${NUM_DUMMIES}_${SEED}.err --job-name=dummy_tests_${CONF}_${NUM_DUMMIES}_${SEED} \
    ./runDummyTests.sh $DATA_DIR $SEED $NUM_DUMMIES $CONF
  done
done
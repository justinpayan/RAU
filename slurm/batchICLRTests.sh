#! /bin/bash

DATA_DIR=/mnt/nfs/scratch1/jpayan/MinimalBidding
LOG_DIR=/mnt/nfs/scratch1/jpayan/logs/MinimalBidding

for SEED in {0..9}; do
  for YEAR in {2018..2022}; do
    sbatch --nodelist=swarm082 --time=00-12:00:00 --partition=defq \
    --nodes=1 --ntasks=1 --output=$LOG_DIR/iclr_tests_${YEAR}_${SEED}.out \
    --error=$LOG_DIR/iclr_tests_${YEAR}_${SEED}.err --job-name=iclr_tests_${YEAR}_${SEED} \
    ./runICLRTests.sh $DATA_DIR $SEED $YEAR
  done
done

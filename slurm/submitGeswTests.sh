#! /bin/bash

DATA_DIR=/mnt/nfs/scratch1/jpayan/RAU
LOG_DIR=/mnt/nfs/scratch1/jpayan/logs/RAU

# ALGO=$1
#for YEAR in {2018..2022}; do
#  sbatch --time=00-11:00:00 --partition=defq \
#  --nodes=1 --ntasks=1 --mem=100G --output=$LOG_DIR/gesw_tests_${YEAR}_${ALGO}.out \
#  --error=$LOG_DIR/gesw_tests_${YEAR}_${ALGO}.err --job-name=gesw_tests_${YEAR}_${ALGO} \
#  ./geswTests.sh $DATA_DIR $YEAR $ALGO
#done

for YEAR in {2018..2022}; do
#  for ALGO in LP GESW; do
#    sbatch --time=00-3:00:00 --partition=defq \
#    --nodes=1 --ntasks=1 --mem=30G --output=$LOG_DIR/gesw_tests_${YEAR}_${ALGO}.out \
#    --error=$LOG_DIR/gesw_tests_${YEAR}_${ALGO}.err --job-name=gesw_tests_${YEAR}_${ALGO} \
#    ./geswTests.sh $DATA_DIR $YEAR $ALGO 0
#  done

#  for ALGO in RRA RRA_ORIG RRA_GESW; do
  ALGO=RRA_ORIG
  for R_IDX in {0..15}; do
    sbatch --time=00-9:00:00 --partition=defq \
    --nodes=1 --ntasks=1 --mem=30G --output=$LOG_DIR/gesw_tests_${YEAR}_${ALGO}_${R_IDX}.out \
    --error=$LOG_DIR/gesw_tests_${YEAR}_${ALGO}_${R_IDX}.err --job-name=gesw_tests_${YEAR}_${ALGO}_${R_IDX} \
    ./geswTests.sh $DATA_DIR $YEAR $ALGO $R_IDX
  done
done
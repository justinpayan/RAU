#! /bin/bash

DATA_DIR=/mnt/nfs/scratch1/jpayan/RAU
LOG_DIR=/mnt/nfs/scratch1/jpayan/logs/RAU


for YEAR in {2018..2023}; do
  sbatch --time=05-11:00:00 --partition=longq \
  --nodes=1 --ntasks=1 --mem=40G --output=$LOG_DIR/gesw_tests_${YEAR}.out \
  --error=$LOG_DIR/gesw_tests_${YEAR}.err --job-name=gesw_tests_${YEAR} \
  ./geswTests.sh $DATA_DIR $YEAR
done

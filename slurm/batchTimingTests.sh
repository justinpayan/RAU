#! /bin/bash

DATA_DIR=/mnt/nfs/scratch1/jpayan/MinimalBidding
LOG_DIR=/mnt/nfs/scratch1/jpayan/logs/MinimalBidding

SEED=0
YEAR=2020
PARTITION=longq
NODELIST="swarm074"
TIME="--time=02-11:59:00"

sbatch $NODELIST $TIME --partition=$PARTITION \
--nodes=1 --ntasks=1 --mem=10G --output=$LOG_DIR/iclr_tests_${YEAR}_${SEED}_true_true.out \
--error=$LOG_DIR/iclr_tests_${YEAR}_${SEED}_true_true.err --job-name=iclr_tests_${YEAR}_${SEED}_true_true \
./runTimingTests.sh $DATA_DIR $SEED $YEAR --caching --dykstra

sbatch $NODELIST $TIME --partition=$PARTITION \
--nodes=1 --ntasks=1 --mem=10G --output=$LOG_DIR/iclr_tests_${YEAR}_${SEED}_true_false.out \
--error=$LOG_DIR/iclr_tests_${YEAR}_${SEED}_true_false.err --job-name=iclr_tests_${YEAR}_${SEED}_true_false \
./runTimingTests.sh $DATA_DIR $SEED $YEAR --caching --no-dykstra

sbatch $NODELIST $TIME --partition=$PARTITION \
--nodes=1 --ntasks=1 --mem=10G --output=$LOG_DIR/iclr_tests_${YEAR}_${SEED}_false_true.out \
--error=$LOG_DIR/iclr_tests_${YEAR}_${SEED}_false_true.err --job-name=iclr_tests_${YEAR}_${SEED}_false_true \
./runTimingTests.sh $DATA_DIR $SEED $YEAR --no-caching --dykstra

sbatch $NODELIST $TIME --partition=$PARTITION \
--nodes=1 --ntasks=1 --mem=10G --output=$LOG_DIR/iclr_tests_${YEAR}_${SEED}_false_false.out \
--error=$LOG_DIR/iclr_tests_${YEAR}_${SEED}_false_false.err --job-name=iclr_tests_${YEAR}_${SEED}_false_false \
./runTimingTests.sh $DATA_DIR $SEED $YEAR --no-caching --no-dykstra


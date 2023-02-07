#! /bin/bash

DATA_DIR=/mnt/nfs/scratch1/jpayan/MinimalBidding
LOG_DIR=/mnt/nfs/scratch1/jpayan/logs/MinimalBidding

SEED=0
YEAR=2020

sbatch --nodelist=swarm082 --time=05-12:00:00 --partition=longq \
--nodes=1 --ntasks=1 --mem=40G --output=$LOG_DIR/iclr_tests_${YEAR}_${SEED}_true_true.out \
--error=$LOG_DIR/iclr_tests_${YEAR}_${SEED}_true_true.err --job-name=iclr_tests_${YEAR}_${SEED}_true_true \
python ../timing_exps.py --data_dir $DATA_DIR --seed $SEED --year $YEAR --caching --dykstra

sbatch --nodelist=swarm082 --time=05-12:00:00 --partition=longq \
--nodes=1 --ntasks=1 --mem=40G --output=$LOG_DIR/iclr_tests_${YEAR}_${SEED}_true_false.out \
--error=$LOG_DIR/iclr_tests_${YEAR}_${SEED}_true_false.err --job-name=iclr_tests_${YEAR}_${SEED}_true_false \
python ../timing_exps.py --data_dir $DATA_DIR --seed $SEED --year $YEAR --caching --no-dykstra

sbatch --nodelist=swarm082 --time=05-12:00:00 --partition=longq \
--nodes=1 --ntasks=1 --mem=40G --output=$LOG_DIR/iclr_tests_${YEAR}_${SEED}_false_true.out \
--error=$LOG_DIR/iclr_tests_${YEAR}_${SEED}_false_true.err --job-name=iclr_tests_${YEAR}_${SEED}_false_true \
python ../timing_exps.py --data_dir $DATA_DIR --seed $SEED --year $YEAR --no-caching --dykstra

sbatch --nodelist=swarm082 --time=05-12:00:00 --partition=longq \
--nodes=1 --ntasks=1 --mem=40G --output=$LOG_DIR/iclr_tests_${YEAR}_${SEED}_false_false.out \
--error=$LOG_DIR/iclr_tests_${YEAR}_${SEED}_false_false.err --job-name=iclr_tests_${YEAR}_${SEED}_false_false \
python ../timing_exps.py --data_dir $DATA_DIR --seed $SEED --year $YEAR --no-caching --no-dykstra


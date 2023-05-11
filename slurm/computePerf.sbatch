#! /bin/bash

DATA_DIR=/mnt/nfs/scratch1/jpayan/MinimalBidding
LOG_DIR=/mnt/nfs/scratch1/jpayan/logs/MinimalBidding

sbatch --time=00-04:00:00 --partition=defq \
--nodes=1 --ntasks=1 --mem=10G --output=$LOG_DIR/computeperf.out \
--error=$LOG_DIR/computeperf.err --job-name=computeperf \
./computePerf.sh $DATA_DIR

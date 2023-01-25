#! /bin/bash

#SBATCH --job-name=iclr_tests
#SBATCH --output=/mnt/nfs/scratch1/jpayan/logs/MinimalBidding/iclr_tests.out
#SBATCH --error=/mnt/nfs/scratch1/jpayan/logs/MinimalBidding/iclr_tests.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=swarm082
#SBATCH --time=06-00:00:00
#SBATCH --partition=longq

DATA_DIR=/mnt/nfs/scratch1/jpayan/MinimalBidding

./runICLRTests.sh $DATA_DIR

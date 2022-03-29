#! /bin/bash

#SBATCH --job-name=repeated_exps_midl_tpms
#SBATCH --output=/mnt/nfs/scratch1/jpayan/repeated_exps_midl_tpms.out
#SBATCH --error=/mnt/nfs/scratch1/jpayan/repeated_exps_midl_tpms.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00-10:00:00
#SBATCH --partition=defq

DATASET=midl
DATA_DIR=/mnt/nfs/scratch1/jpayan/MinimalBidding
QUERY_MODEL=tpms

./repeatedExps.sh $DATASET $DATA_DIR $QUERY_MODEL

#! /bin/bash

#SBATCH --job-name=midl_solver_tpms
#SBATCH --output=/mnt/nfs/scratch1/jpayan/midl_solver_tpms.out
#SBATCH --error=/mnt/nfs/scratch1/jpayan/midl_solver_tpms.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=swarm082
#SBATCH --mem=10G
#SBATCH --time=00-10:00:00
#SBATCH --partition=defq

DATASET=midl
DATA_DIR=/mnt/nfs/scratch1/jpayan/MinimalBidding
QUERY_MODEL=tpms

./submitSolver.sh $DATASET $DATA_DIR $QUERY_MODEL

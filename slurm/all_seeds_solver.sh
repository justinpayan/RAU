#!/usr/bin/env bash

./submitSolver.sbatch midl random
./submitSolver.sbatch cvpr random
./submitSolver.sbatch cvpr18 random
./submitSolver.sbatch midl tpms
./submitSolver.sbatch cvpr tpms
./submitSolver.sbatch cvpr18 tpms
./submitSolver.sbatch midl superstar
./submitSolver.sbatch cvpr superstar
./submitSolver.sbatch cvpr18 superstar


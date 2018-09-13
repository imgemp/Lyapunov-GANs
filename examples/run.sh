#!/bin/bash

export MKL_NUM_THREADS=3
export OPENBLAS_NUM_THREADS=3
export OMP_NUM_THREADS=3

set -exu

input=$1
# echo $PWD
# -m torch.utils.bottleneck
# -m memory_profiler
PYTHONPATH=./ python lyapunov/run.py $(cat $input)

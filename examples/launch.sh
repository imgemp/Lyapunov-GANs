#!/bin/bash

export MKL_NUM_THREADS=3
export OPENBLAS_NUM_THREADS=3
export OMP_NUM_THREADS=3

set -exu

# configs=( "args/exp1.txt" "args/exp2.txt" "args/exp3.txt" "args/exp4.txt" "args/exp5.txt" )

# for c in ${configs[@]}
for i in $(seq 0 9); do
	if [ $i -lt 10 ]; then
		sbatch -p m40-long --gres=gpu:1 examples/run.sh "examples/args/CIFAR10/cc/0"$i".txt"
	else
		sbatch -p m40-long --gres=gpu:1 examples/run.sh "examples/args/CIFAR10/cc/"$i".txt"
	fi
done

for i in $(seq 0 9); do
    if [ $i -lt 10 ]; then
        sbatch -p m40-long --gres=gpu:1 examples/run.sh "examples/args/CIFAR10/con/0"$i".txt"
    else
        sbatch -p m40-long --gres=gpu:1 examples/run.sh "examples/args/CIFAR10/con/"$i".txt"
    fi
done

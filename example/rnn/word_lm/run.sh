#!/usr/bin/env bash
#
# Usage: 
# Author: viking(auimoviki@gmail.com)

#SBATCH -p 4gpuq/airesearch --gres gpu:1 --mem 20G -J lm -o ./log/softmax.log

eval -- set "./data/ptb/ 10 ./exp/debug 1"

if [[ $# -lt 4 ]]; then
    echo 'USAGE: data numlab outdir use_nce'
    exit 0
fi

module add wxm71/mxnet/python3/master 

data=$1
numlab=$2
outdir=$3
use_nce=$4

if [[ 1 == ${use_nce} ]]; then
    opt='--use-nce'
else 
    opt=''
fi

mkdir -p ${outdir}

srun -p 4gpuq --gres gpu:1 --mem 20G -J lm \
    python3 train.py --tied --nhid 650 --emsize 650 --output ${outdir} --lr 0.01 \
	--num-label ${numlab} --data ${data} --log-interval 100 --batch_size 32 ${opt}

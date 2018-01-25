#!/usr/bin/env bash
#
# Usage: 
# Author: wxm71(weixing.mei@aispeech.com)

#SBATCH -p 4gpuq/airesearch --gres gpu:1 --mem 10G -J lm -o ./log/softmax.log

module add wxm71/mxnet/python3/master 

outdir=./exp/nce
mkdir -p ${outdir}
rm -rf ${outdir}/*

python3 train.py --tied --nhid 650 --emsize 650 --output ${outdir} --lr 0.01 --num-label 10 \
    --data ./data/debug/ --log-interval 10 --batch_size 1 --use-nce

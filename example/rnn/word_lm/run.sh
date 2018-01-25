#!/usr/bin/env bash
#
# Usage: 
# Author: wxm71(weixing.mei@aispeech.com)

#SBATCH -p 4gpuq/airesearch --gres gpu:1 --mem 10G -J lm -o ./log/softmax.log

module add wxm71/mxnet/python3/master 

outdir=./exp/softmax
mkdir -p ${outdir}
rm -rf ${outdir}/*

python3 train.py --tied --nhid 650 --emsize 650 --output ${outdir}

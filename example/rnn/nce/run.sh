#!/usr/bin/env bash
#
# Usage: 
# Author: wxm71(weixing.mei@aispeech.com)

#SBATCH --gres gpu:1 --mem 10G -p 4gpuq -o log/ce.log

module add mxnet/python3/0.12.1

outdir=./exp/ce
mkdir -p ${outdir}

srun --gres gpu:1 --mem 10G -p 4gpuq \
    python3 ./src/lstm_word.py --output ${outdir} --gpu --lr 0.01

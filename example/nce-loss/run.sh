#!/usr/bin/env bash
#
# Usage: 
# Author: wxm71(weixing.mei@aispeech.com)

#SBATCH -p 4gpuq/airesearch --gres gpu:1 --mem 10G -J lm -o ./exp/raw/train.log

module add wxm71/mxnet/python3/master 

python3 lstm_word.py --gpu | tee run.log

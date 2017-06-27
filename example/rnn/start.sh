#!/usr/bin/env bash
#
# Usage: 
# Author: Summer Qing(qingyun.wu@aispeech.com)

# set mxnet to current version instead of system install version 
source ./.bashrc

mkdir -p ./output/{gradient,logs,model}
rm -rf ./output/*/* 

python ./src/cudnn_lstm_nce.py --model-prefix ./output/model/lstm \
    --disp-batches 40 \
    --num-label 50 --lr 0.01 --batch-size 40 \
    --wd 1e-5 \
    --gpu 3 \
    "$@"

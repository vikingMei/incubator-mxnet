#!/usr/bin/env bash
#
# Usage: 
# Author: Summer Qing(qingyun.wu@aispeech.com)

source ./.bashrc

rm -rf ./output/*/* 

python ./src/cudnn_lstm_nce.py --model-prefix ./output/model/lstm \
    --disp-batches 40 \
    --gpus 1 --num-label 50 --lr 0.01 --batch-size 40 "$@"

#!/usr/bin/env bash
#
# Usage: 
# Author: Summer Qing(qingyun.wu@aispeech.com)

# set mxnet to current version instead of system install version 
source ./.bashrc

mkdir -p ./output/{gradient,logs,model}

prefix=./output/model

rm -rf ./output/gradient/* 
rm -rf ./output/logs/*
rm ${prefix}/*

python ./src/cudnn_lstm_nce.py --model-prefix ${prefix}/lstm \
    --disp-batches 40 \
    --num-label 10 --lr 0.1 --batch-size 40 \
    --gpus 1 \
    --wd 1e-5 \
    --min-epoch 1 \
    "$@" 

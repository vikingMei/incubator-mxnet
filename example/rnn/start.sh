#!/usr/bin/env bash
#
# Usage: 
# Author: Summer Qing(qingyun.wu@aispeech.com)

source ./.bashrc

rm ./output/*/* -rf

python ./src/cudnn_lstm_nce.py --gpus 1 --num-label 5 --lr 0.001 --batch-size 1 "$@"

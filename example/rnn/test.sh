#!/usr/bin/env bash
#
# Usage: 
# Author: Summer Qing(qingyun.wu@aispeech.com)

source .bashrc
python  ./src/cudnn_lstm_nce.py --test \
    --model-prefix=./output/raw/lstm --gpus 0 \
    --batch-size 10 \
    --load-epoch "$@"

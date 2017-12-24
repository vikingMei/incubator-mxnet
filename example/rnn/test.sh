#!/usr/bin/env bash
#

source .bashrc
python  ./src/cudnn_lstm_nce.py --test \
    --model-prefix=./output/model/lstm --gpus 0 \
    --batch-size 10 \
    --load-epoch "$@"

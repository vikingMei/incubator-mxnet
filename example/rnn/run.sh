#!/usr/bin/env bash
#
# Usage: 
# Author: weixing.mei(auimoviki@gmail.com)

args=`getopt -o 'd:' "$@"`

mod='train'

if [[ $# -gt 0 ]]; then
    mod=$1
fi

if [[ $# -gt 1 ]]; then
    debug=$2
fi

if [[ ${mod} = 'test' ]]; then
    MODS='--test --load-epoch 1 --batch-size 1'
elif [[ ${mod} = 'train' ]]; then
    MODS=''
else
    echo "invalid mod: [${mod}], only 'test' or 'train'(default) support"
    exit 0
fi

DEBUG='-m pdb'
DEBUG=''
SRC=./src/lstm_bucketing.py
SRC=./src/cudnn_lstm_nce.py
python ${DEBUG} ${SRC}  --stack-rnn False --batch-size 20 --num-label 5 \
    --train-data ./data/train.txt \
    --valid-data ./data/valid.txt \
    --test-data  ./data/ptb.test.txt \
    ${MODS} \
    --model-prefix ./model/lstm 

#dot -Tpdf -o plot.pdf ./plot.gv

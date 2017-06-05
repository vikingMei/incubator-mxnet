#!/usr/bin/env bash
#
# Usage: 
# Author: weixing.mei(auimoviki@gmail.com)

function usage() {
    echo "USAGE: $0 [-d] [train|test]"
    echo ""
    echo "PARAMETER: "
    echo "  -d: debug mod, start with python pdb"
    echo "  train: run in train mode, default"
    echo "  test:  run in test mode"

    exit 0
}

args=`getopt -o 'd:' -- "$@"`
if [[ $? != 0 ]]; then
    usage
fi

eval set -- "${args}"

DEBUG=""
while true;
do
    case $1 in 
	-d|--debug) DEBUG="-m pdb"; shift 1;;
	--) shift; break;;
	*)  usage;;
    esac
done

mod='train'
if [[ $# -gt 0 ]]; then
    mod=$1
fi

if [[ ${mod} = 'test' ]]; then
    MOD='--test --load-epoch 18 --batch-size 1'
elif [[ ${mod} = 'train' ]]; then
    MOD='--batch-size 20'
else
    echo "invalid mod: [${mod}], only 'test' or 'train'(default) support"
    exit 0
fi

echo "MOD: ${MOD}"
echo "DEBUG: ${DEBUG}"

SRC=./cudnn_lstm_nce.py
python ${DEBUG} ${SRC}  ${MOD} \
    --stack-rnn False --num-label 5 \
    --train-data ./data/ptb.train.txt \
    --valid-data ./data/ptb.valid.txt \
    --test-data  ./data/ptb.test.txt \
    --model-prefix ./model/lstm 

#dot -Tpdf -o plot.pdf ./plot.gv

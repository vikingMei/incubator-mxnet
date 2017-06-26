#!/usr/bin/env bash
#
# Usage: 
# Author: Summer Qing <qingyun.wu@aispeech.com>

source ./.bashrc
rm -rf ./output/logs/*
#rm -rf ./gradient/*
#ps aux | grep python | grep cudnn_lstm_nce | awk '{print $2}' | xargs kill -s SIGKILL 

function usage() {
    echo "USAGE: $0 [-d] [-g] [train|test]"
    echo ""
    echo "PARAMETER: "
    echo "  -d:         debug mod, start with python pdb"
    echo "  -g:         enable gpu or not"
    echo "  -e epoch    start from epoch, default start from scratch"
    echo "  -b size     batch size, default 40"
    echo "  -n num      num of label, default 5"
    echo "  -l rate     learning rate"
    echo "  train: run in train mode, default"
    echo "  test:  run in test mode"

    exit 0
}

args=`getopt -o 'e:b:n:l:dg' -- "$@"`
if [[ $? != 0 ]]; then
    usage
fi

eval set -- "${args}"

DEBUG=""
GPU=""
EPOCH=""
BATCH="--batch-size 40"
LABEL="num-label 5"
LRATE="--lr 0.001"
while true;
do
    case $1 in 
        -e|--epoch) EPOCH="--load-epoch $2"; shift 2;;
	-d|--debug) DEBUG="-m pdb"; shift 1;;
        -g|--gpu)   GPU="--gpus 1"; shift 1;;
        -b|--batch) BATCH="--batch-size $2"; shift 2;;
        -n|--label) LABEL="--num-label $2"; shift 2;;
        -l|--lr)    LRATE="--lr $2"; shift 2;;
	--) shift; break;;
	*)  usage;;
    esac
done

mod='train'
if [[ $# -gt 0 ]]; then
    mod=$1
fi

if [[ ${mod} = 'test' ]]; then
    MOD='--test'
elif [[ ${mod} = 'train' ]]; then
    MOD='--num-epochs 30'
else
    echo "invalid mod: [${mod}], only 'test' or 'train'(default) support"
    exit 0
fi

echo "args: " ${DEBUG} ${SRC} ${EPOCH} ${GPU} ${MOD} ${BATCH} ${LABEL} ${LRATE}

SRC=./src/cudnn_lstm_nce.py

#gdb -x ./gdb.cmds --args \
python ${DEBUG} ${SRC} ${EPOCH} ${GPU} ${MOD} ${BATCH} ${LABEL} ${LRATE} \
    --lr 0.01 \
    --disp-batch 40 \
    --train-data ./data/ptb.train.txt \
    --valid-data ./data/ptb.valid.txt \
    --test-data  ./data/ptb.test.txt \
    --model-prefix ./output/model/lstm \
    "$@"

#dot -Tpdf -o plot.pdf ./plot.gv

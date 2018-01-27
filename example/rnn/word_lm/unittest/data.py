#!/usr/bin/env python3
# coding: utf-8
#
# Usage: 
# Author: wxm71(weixing.mei@aispeech.com)

import sys
sys.path.append('.')

import logging

from nce import NceCorpus

logging.basicConfig(level=logging.DEBUG)

batch_size = 32 
bptt = 35 
num_label = 100

rootdir='./data/ptb/'
corpus = NceCorpus('%s/train.txt'%rootdir, '%s/valid.txt'%rootdir, '%s/test.txt'%rootdir)

train_iter = corpus.get_test_iter(batch_size, bptt, num_label, 10)

idx = 0
for data in train_iter:
    print(data)
    idx += 1


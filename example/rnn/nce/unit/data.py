#!/usr/bin/env python3
# coding: utf-8
#
# Usage: 
# Author: wxm71(weixing.mei@aispeech.com)

import pdb
import sys
import logging

sys.path.append('./src/')
from loader import Corpus,CorpusIter

logging.basicConfig(level=logging.DEBUG)

corpus = Corpus(basedir='./data/ptb/')
test_iter = corpus.get_test_iter(64, 35, 1)

for batch in test_iter:
    print(batch)
    pdb.set_trace()

#!/usr/bin/env python
# coding: utf-8
#
# Usage: 
# Author: wxm71(weixing.mei@aispeech.com)

import math
import logging
import mxnet as mx

from .utils import tokenize, batchify
from .vocab import Vocab
from .corpusiter import CorpusIter


class Corpus(object):
    def __init__(self, basedir=None, vocab:Vocab=None):
        self.logger = logging.getLogger(str(self.__class__))

        ftest = '%s/test.txt'  % basedir
        ftrain = '%s/train.txt' % basedir
        fvalid = '%s/valid.txt' % basedir

        update_vocab = vocab is None
        self.data_train, self.vocab = tokenize(ftrain, vocab, update_vocab=update_vocab, eos=True)
        self.data_valid, _ = tokenize(fvalid, vocab, eos=True)
        self.data_test, _ = tokenize(ftest, vocab, eos=True)

        self._test_iter = None
        self._train_iter = None
        self._valid_iter = None

        self.wrdfrq = []
        self.total_wrd = 0

        self.build_wrdcnt()


    def build_wrdcnt(self):
        self.total_wrd = 0
        self.wrdfrq = [0.0]*len(self.vocab)

        for idx in self.data_train: 
            self.wrdfrq[idx] += 1
            self.total_wrd += 1

        for idx,cnt in enumerate(self.wrdfrq): 
            self.wrdfrq[idx] /= self.total_wrd


    def _get_iter(self, data, batch_size, bptt, num_parall=2):
        return CorpusIter(data, batch_size, bptt, num_parall) 


    def get_train_iter(self, batch_size, bptt, num_parall=2):
        if not self._train_iter:
            self._train_iter = self._get_iter(self.data_train, batch_size, bptt, num_parall) 
        return self._train_iter


    def get_test_iter(self, batch_size, bptt, num_parall=2):
        if not self._test_iter:
            self._test_iter = self._get_iter(self.data_test, batch_size, bptt, num_parall) 
        return self._test_iter


    def get_valid_iter(self, batch_size, bptt, num_parall=2):
        if not self._valid_iter:
            self._valid_iter = self._get_iter(self.data_valid, batch_size, bptt, num_parall) 
        return self._valid_iter

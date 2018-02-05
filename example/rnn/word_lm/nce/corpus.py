#!/usr/bin/env python
# coding: utf-8
#
# Usage: 
# Author: viking(auimoviki@gmail.com)

import math
import logging
import mxnet as mx

from .utils import tokenize, Vocab
from .corpusiter import NceCorpusIter


class NceCorpus(object):
    def __init__(self, ftrain, ftest, fvalid, vocab:Vocab=None):
        self.logger = logging.getLogger('NceCorpus') 

        update_vocab = vocab is None
        self.data_train, self.vocab = tokenize(ftrain, vocab, update_vocab=update_vocab, eos=True)
        self.data_valid, _ = tokenize(fvalid, vocab, eos=True)
        self.data_test, _ = tokenize(ftest, vocab, eos=True)

        self._test_iter = None
        self._train_iter = None
        self._valid_iter = None

        self.wrdfrq = []
        self.negdis = []
        self.negative = []

        self.build_negative()


    def build_negative(self):
        total_wrd = 0

        # build negdis for train corpus
        self.wrdfrq = [0.0]*len(self.vocab)
        for idx in self.data_train: 
            self.wrdfrq[idx] += 1
            total_wrd += 1

        total_cnt = 0
        self.negdis = [0]*len(self.vocab)
        for idx,cnt in enumerate(self.wrdfrq):
            self.wrdfrq[idx] /= total_wrd
            if idx<Vocab.FIRST_VALID_ID or cnt<5:
                self.negdis[idx] = 0.0
            else:
                v = int(math.pow(cnt, 0.75))
                self.negdis[idx] = v
                self.negative.extend([idx]*v)

                total_cnt += v

        denorm = float(total_cnt)
        for key,_ in enumerate(self.negdis):
            self.negdis[key] /= denorm
        self.negdis = mx.nd.array(self.negdis)


    def _get_iter(self, data, batch_size, bptt, numlab, num_parall=2):
        return NceCorpusIter(data, batch_size, bptt, numlab, self.negative, num_parall) 


    def get_train_iter(self, batch_size, bptt, numlab, num_parall=2):
        if not self._train_iter:
            self._train_iter = self._get_iter(self.data_train, batch_size, bptt, numlab, num_parall) 
        return self._train_iter


    def get_test_iter(self, batch_size, bptt, numlab, num_parall=2):
        if not self._test_iter:
            self._test_iter = self._get_iter(self.data_test, batch_size, bptt, numlab, num_parall) 
        return self._test_iter


    def get_valid_iter(self, batch_size, bptt, numlab, num_parall=2):
        if not self._valid_iter:
            self._valid_iter = self._get_iter(self.data_valid, batch_size, bptt, numlab, num_parall) 
        return self._valid_iter

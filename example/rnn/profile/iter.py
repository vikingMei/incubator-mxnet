#!/usr/bin/env python
# coding: utf-8
#
# Usage: 

import sys
import time
import mxnet as mx

from utils import tokenize_text
from nce import LMNceIter

def time_eval(func):
    tic = time.time()
    func()
    print 'time: %f' % (time.time()-tic)

def raw_iter(sent, batch_size, buckets, invlab):
    tic = time.time()
    dataIter  = mx.rnn.BucketSentenceIter(sent, batch_size, buckets=buckets, invalid_label=invlab)
    print 'time: %f' % (time.time()-tic)

    tic = time.time()
    for item in dataIter:
        continue

    mx.ndarray.waitall()
    print 'time: %f' % (time.time()-tic)


def nce_iter(sent, batch_size, buckets, invlab, freq, layout, numlab):
    tic = time.time()
    dataIter  = LMNceIter(sent, batch_size, freq, 
                          layout=layout, buckets=buckets, 
                          pad_label=invlab, num_label=numlab)
    print 'time: %f' % (time.time()-tic)

    tic = time.time()
    for item in dataIter:
        continue

    mx.ndarray.waitall()
    print 'time: %f' % (time.time()-tic)


def nce_iter_disp(sent, batch_size, buckets, invlab, freq, layout, numlab):
    dataIter  = LMNceIter(sent, batch_size, freq, 
                          layout=layout, buckets=buckets, 
                          invalid_label=invlab, num_label=numlab)

    for item in dataIter:
        print item.data[0].asnumpy()
        print item.label[0].asnumpy()
        print item.label[1].asnumpy()
        break


if "__main__"==__name__:
    reload(sys)
    sys.setdefaultencoding('utf-8')

    fname = './data/ptb.train.txt'
    sent,vocab,freq = tokenize_text(fname, invalid_label=0, start_label=1)

    batch_size = 20
    buckets = [10, 20, 30, 40, 50, 60, 70, 80]
    invlab=0

    # time: 0.244546
    # time: 0.037611
    # time: 0.021864
    #raw_iter(sent, batch_size, buckets, invlab)

    numlab = 50
    nce_iter(sent, batch_size, buckets, invlab, freq, "NT", numlab)

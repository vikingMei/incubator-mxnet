#!/usr/bin/env python
# coding: utf-8
#
# Usage: 

import os
import sys
import mxnet as mx

from toy_data import gen_toy_data

def iter_content():
    dataIter  = gen_toy_data()
    print 'lines: ', dataIter.raw_data
    print 'sent: ', dataIter.sent
    print 'vocab: ', dataIter.vocab
    print 'freq: ', dataIter.freq

    dataIter.reset()
    for item in dataIter:
        print item.data[0].asnumpy()
        print item.label[0].asnumpy()
        print item.label[1].asnumpy()
        print "\n"*2

    #os.remove(fname)


if "__main__"==__name__:
    reload(sys)
    sys.setdefaultencoding('utf-8')

    iter_content()

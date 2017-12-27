#!/usr/bin/env python3
# coding: utf-8
#
# Usage: 
# Author: wxm71(weixing.mei@aispeech.com)

import sys
import module
import mxnet as mx
from data import Corpus, CorpusIter


def main():
    """
    main function
    """
    ctx = mx.gpu(2)

    batch_size = 40
    bptt = 35

    corpus = Corpus('./data/ptb.')
    ntokens = len(corpus.dictionary)
    train_data = CorpusIter(corpus.train, batch_size, bptt)
    valid_data = CorpusIter(corpus.valid, batch_size, bptt)
    test_data = CorpusIter(corpus.test, batch_size, bptt)

    data_names = [x[0] for x in test_data.provide_data] 
    label_names = [x[0] for x in test_data.provide_label]

    prefix = './output/model'
    epoch = 39
    model = mx.module.Module.load(prefix, epoch, label_names=label_names, data_names=data_names, context=ctx)
    model.bind(for_training=False, data_shapes=test_data.provide_data, label_shapes=test_data.provide_label)

    for batch in test_data:
        model.forward(batch, is_train=False)
        print(model.get_outputs())


if "__main__"==__name__:
    main()

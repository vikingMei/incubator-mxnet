#!/usr/bin/env python
# coding: utf-8
#
# Usage: 

import os
from utils import tokenize_text
from nce import LMNceIter

def gen_toy_data(batch_size=1, pad_label=0, invalid_lab=1, start_label=2, num_label=5):
    buckets = [2, 5]
    lines = [ "this ", "line 1 ", "line 2 ", "line one two ", "a b c "]

    fname = os.tmpnam()
    with open(fname, 'w') as fid:
        fid.write('\n'.join(lines))
        fid.write('\n')

    sent, vocab, freq = tokenize_text(fname, vocab=None, invalid_label=1, start_label=2) 

    dataIter  = LMNceIter(sent, batch_size, freq, layout='NT', buckets=buckets, pad_label=pad_label, num_label=num_label)

    dataIter.sent = sent
    dataIter.freq = freq
    dataIter.vocab = vocab
    dataIter.raw_data = lines

    dataIter.pad_label = pad_label
    dataIter.start_label = start_label
    dataIter.invalid_label = invalid_lab

    return dataIter

def gen_train_data(batch_size, num_label):
    buckets = [10, 20, 30, 40, 50, 60, 70, 80]

    sent, vocab, freq = tokenize_text("./data/train.txt", start_label=2, invalid_label=1)
    assert None==vocab.get(''), "'' shouldn't appeare in sentences"
    vocab[''] = 0

    # layout, format of data and label. 'NT' means (batch_size, length) and 'TN' means (length, batch_size).
    dataIter  = LMNceIter(sent, batch_size, freq, layout='NT', buckets=buckets, pad_label=0, num_label=5)

    dataIter.raw_data = ''
    dataIter.sent = sent
    dataIter.freq = freq
    dataIter.vocab = vocab

    dataIter.pad_label = 0
    dataIter.start_label = 2
    dataIter.invalid_label = 1
    return dataIter


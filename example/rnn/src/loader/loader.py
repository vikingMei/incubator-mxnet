#!/usr/bin/env python
# coding: utf-8
#
# Usage: 
# Author: Summer Qing(qingyun.wu@aispeech.com)

import json
import mxnet as mx
from lstm_nce import LMNceIter 


def tokenize_text(fname, vocab=None, invalid_label=-1, start_label=0):
    """
    translate input text to id list

    RETURN:
        sentences: list of id list
        vocab: dict, word to id map 
        freq: dict, frequence of each word
    """
    # read whole file, ans split each line into an word array
    lines = open(fname).readlines()
    lines = [['<s>'] +filter(None, i.split(' ')) for i in lines]

    # map word list into id list
    sentences, vocab = mx.rnn.encode_sentences(lines, vocab=vocab, invalid_label=invalid_label, start_label=start_label)

    # get frequence of eacho word 
    freq = {}
    for line in sentences:
        for val in line:
            if val not in freq:
                freq[val] = 1
            else:
                freq[val] += 1

    return sentences, vocab, freq



def get_nce_iter(fname, start_label, invalid_label, pad_label, batch_size, buckets, num_label, vocab=None, freq=None, rand=True):
    '''
    get data iter for nce train
    '''
    layout = 'NT'
    if vocab is None:
        # for train
        sent, vocab, freq = tokenize_text(fname, start_label=start_label, invalid_label=invalid_label)

        assert None==vocab.get(''), "'' shouldn't appeare in sentences"
        vocab[''] = pad_label
        freq[str(pad_label)] = 0

        with open('./output/vocab.json', 'w') as fid:
            fid.write(json.dumps(vocab))

        with open('./output/freq.json', 'w') as fid:
            fid.write(json.dumps(freq))
    else:
        # NOTE: in this function, will encode word that not in vocab build from train set and extend vocab, 
        # which may be undesired
        sent, _, _ = tokenize_text(fname, vocab=vocab, start_label=start_label, invalid_label=invalid_label)

    # layout, format of data and label. 'NT' means (batch_size, length) and 'TN' means (length, batch_size).
    dataiter  = LMNceIter(sent, batch_size, freq, layout=layout, buckets=buckets, pad_label=pad_label, num_label=num_label, rand=rand)

    return dataiter, vocab, freq

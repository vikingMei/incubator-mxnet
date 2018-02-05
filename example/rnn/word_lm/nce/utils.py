#!/usr/bin/env python3
# coding: utf-8
#
# Usage: 
# Author: viking(auimoviki@gmail.com)

from typing import List
from .vocab import Vocab

def tokenize(fname, vocab, update_vocab=False, bos=False, eos=False):
    '''
    tokenize given file
    '''
    if not vocab:
        vocab = Vocab()
        update_vocab = True

    ids = []
    with open(fname, 'r') as fid:
        for line in fid:
            arr = line.split()

            if bos: ids.append(Vocab.BOS_ID) 

            for wrd in arr:
                wrd = wrd.strip()
                wid = vocab.add_wrd(wrd) if update_vocab else vocab.get_wrd(wrd)
                ids.append(wid)

            if eos: ids.append(Vocab.EOS_ID)
    return ids, vocab


def batchify(data, batch_size):
    """Reshape data into (num_example, batch_size)"""
    nbatch = data.shape[0] // batch_size
    data = data[:nbatch * batch_size]
    data = data.reshape((batch_size, nbatch)).T
    return data



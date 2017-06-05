#!/usr/bin/env python
# coding: utf-8
#
# Usage: 
# Author: weixing.mei(auimoviki@gmail.com)

import mxnet as mx

def tokenize_text(fname, vocab=None, invalid_label=-1, start_label=0):
    """
    translate input text to id list

    RETURN:
        sentences: list of id list
        vocab: dict, word to id map 
        freq: list, frequence of each word
    """
    # read whole file, ans split each line into an word array
    lines = open(fname).readlines()
    lines = [filter(None, i.split(' ')) for i in lines]

    # map word list into id list
    sentences, vocab = mx.rnn.encode_sentences(lines, vocab=vocab, invalid_label=invalid_label, start_label=start_label)

    # get frequence of eacho word 
    freq = [0]*len(vocab)
    for line in sentences:
        for val in line:
            freq[val] += 1

    return sentences, vocab, freq


#!/usr/bin/env python
# coding: utf-8
#
# Usage: 
# Author: wxm71(weixing.mei@aispeech.com)

import os
import json

class Vocab(object):
    '''
    wrd dictionary
    '''
    PAD = '<pad>'
    UNK = '<unk>'
    BOS = '<s>'
    EOS = '</s>'

    PAD_ID = 0
    UNK_ID = 1
    BOS_ID = 2
    EOS_ID = 3

    FIRST_VALID_ID = EOS_ID+1

    def __init__(self):
        self.wrd2idx = {}
        self.idx2wrd = []

        self.add_wrd(Vocab.PAD)
        self.add_wrd(Vocab.UNK)
        self.add_wrd(Vocab.BOS)
        self.add_wrd(Vocab.EOS)

    def __len__(self):
        return len(self.wrd2idx) 


    def get_wrd(self, wrd):
        return self.wrd2idx.get(wrd) or Vocab.UNK_ID


    def add_wrd(self, wrd):
        if wrd not in self.wrd2idx:
            self.idx2wrd.append(wrd)
            self.wrd2idx[wrd] = len(self.idx2wrd) - 1
        return self.wrd2idx[wrd]


    def dump(self, fname, ffreq=None):
        fid = open(fname, 'w')
        json.dump(self.wrd2idx, fid)
        fid.close()


    def load_vocab(self, fvocab, ffreq=None):
        assert os.path.exists(fvocab)

        fid = open(fname, 'r')
        self.wrd2idx = json.load(fid)
        fid.close()
        self.idx2wrd = [0]*len(self.idx2wrd)
        for k,v in self.wrd2idx.items(): 
            self.idx2wrd[v] = k


    def __len__(self):
        return len(self.idx2wrd)

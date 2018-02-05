#!/usr/bin/env python
# coding: utf-8
#
# Usage: 
# Author: wxm71(weixing.mei@aispeech.com)

import random
import logging
import numpy as np
import mxnet as mx

import pdb


class NceLabGenerator():
    def __init__(self, data, idxbeg, idxend, batch_size, bptt, numlab, negative, queue):
        self.logger = logging.getLogger('NceLabGenerator')

        self.data = data 
        self.bptt = bptt 
        self.batch_size = batch_size

        self.idxbeg = idxbeg
        self.idxend = idxend

        self.numlab = numlab
        self.negative = negative

        self.idx = idxbeg
        self.queue = queue


    def simple_sample(self):
        '''
        sample label for nce train, negative label may equal to positive label
        '''
        # [1, bptt,batch_size]
        tmp1 = self.data[self.idx+1:self.idx+1+self.bptt]
        tmp1 = tmp1.reshape((1, self.bptt, self.batch_size))

        # [numlab-1, bptt, batch_size]
        if self.numlab>1:
            tmp2 = np.zeros((self.numlab-1)*self.bptt*self.batch_size) 
            for i in range(0,tmp2.size):
                tmp2[i] = self.negative[random.randint(0, len(self.negative) - 1)] 
            tmp2 = tmp2.reshape((self.numlab-1, self.bptt, self.batch_size))
            return np.concatenate((tmp1, tmp2), axis=0).transpose((1,2,0))
        else:
            return tmp1.transpose((1,2,0)) 


    def norepeat_sample(self):
        '''
        sample label for nce train, without negative label same with positive label
        '''
        numneg = len(self.negative) 

        # [bptt,batch_size,numlab]
        label = np.zeros((self.bptt, self.batch_size, self.numlab))
        for i in range(0, self.bptt):
            for j in range(0, self.batch_size):
                v = self.data[self.idx+i+1,j]
                label[i,j,0] = v
            
                for k in range(1, self.numlab):
                    neg = v
                    while v==neg:
                        neg = self.negative[random.randint(0, numneg-1)]
                    label[i,j,k] = neg 
        return label


    def run(self):
        label_weight = np.zeros((self.bptt, self.batch_size, self.numlab), dtype='float32')
        label_weight[:,:,0] = 1.0
        label_weight = mx.nd.array(label_weight)

        idxend = self.idxend-self.bptt
        while self.idx<idxend:
            data = self.data[self.idx:self.idx+self.bptt,:] 
            label = self.norepeat_sample()
            self.idx += self.bptt

            data = mx.nd.array(data, dtype='int32')
            label = mx.nd.array(label, dtype='int32')
            self.queue.put((data, label, label_weight))

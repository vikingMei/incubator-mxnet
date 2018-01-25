# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import os
import sys
import pdb
import math
import gzip
import json
import time
import random
import mxnet as mx
import numpy as np

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.word_count = []
        self.add_word('<unk>')
        self.add_word('<eos>')

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
            self.word_count.append(0)
        index = self.word2idx[word]
        self.word_count[index] += 1
        return index

    def dump(self, fname):
        fid = open(fname, 'w')
        json.dump(self.word2idx, fid)
        fid.close()

    def __len__(self):
        return len(self.idx2word)

class Corpus(object):
    def __init__(self, path, is_train=False):
        self.dictionary = Dictionary()
        self.train = self.tokenize(path + 'train.txt')
        self.valid = self.tokenize(path + 'valid.txt')
        self.test = self.tokenize(path + 'test.txt')

        self.negative = []  

        negdis = [1]*len(self.dictionary.word_count)
        for idx,freq in enumerate(self.dictionary.word_count):
            # skip <unk> and <eos>
            if idx<2 or freq<5:
                negdis[idx] = 0
                continue
            v = int(math.pow(freq * 1.0, 0.75))
            negdis[idx] = v
            self.negative.extend([idx]*v)
        
        self.negdis = mx.nd.array(negdis,dtype='float32')
        tmp = self.negdis.sum()
        self.negdis = self.negdis/tmp


    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        ids = []
        with open(path, 'r') as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    wid = self.dictionary.add_word(word)
                    ids.append(wid)

        return mx.nd.array(ids, dtype='int32')


def batchify(data, batch_size):
    """Reshape data into (num_example, batch_size)"""
    nbatch = data.shape[0] // batch_size
    data = data[:nbatch * batch_size]
    data = data.reshape((batch_size, nbatch)).T
    return data


class CorpusIter(mx.io.DataIter):
    "An iterator that returns the a batch of sequence each time"
    def __init__(self, source, batch_size, bptt):
        super(CorpusIter, self).__init__()
        self.batch_size = batch_size
        self.provide_data = [('data', (bptt, batch_size), np.int32)]
        self.provide_label = [('label', (bptt, batch_size))]
        self._index = 0
        self._bptt = bptt
        self._source = batchify(source, batch_size)


    def iter_next(self):
        i = self._index
        if i+self._bptt > self._source.shape[0] - 1:
            return False
        self._next_data = self._source[i:i+self._bptt]
        self._next_label = self._source[i+1:i+1+self._bptt].astype(np.float32)
        self._index += self._bptt
        return True


    def next(self):
        if self.iter_next():
            return mx.io.DataBatch(data=self.getdata(), label=self.getlabel())
        else:
            raise StopIteration

    def reset(self):
        self._index = 0
        self._next_data = None
        self._next_label = None

    def getdata(self):
        return [self._next_data]

    def getlabel(self):
        return [self._next_label]


class NceCorpusIter(CorpusIter):
    def __init__(self, source, batch_size, bptt, num_label, negative):
        super(NceCorpusIter, self).__init__(source, batch_size, bptt)
        self.num_label = num_label
        self.negative = negative
        label_shape = (bptt, batch_size, num_label)
        self.provide_label = [('label', label_shape, np.int32), ('label_weight', label_shape)]
        self.provide_data = [('data', (bptt, batch_size), np.int32)]

    def getlabel(self):
        return [*self._next_label]

    def iter_next(self):
        i = self._index
        if i+self._bptt > self._source.shape[0] - 1:
            return False
        self._index += self._bptt

        # generate data
        self._next_data = self._source[i:i+self._bptt]

        # generate label
        label_shape = (self.num_label, self._bptt, self.batch_size)
        label = mx.nd.zeros(label_shape)

        # [bptt,batch_size]
        #label = np.zeros((self._bptt, self.batch_size, self.num_label))
        #for i in range(0, self._bptt):
        #    for j in range(0, self.batch_size):
        #        v = self._source[i+1,j].asnumpy()
        #        label[i,j,0] = v
        #    
        #        for k in range(1, self.num_label):
        #            neg = v
        #            while v==neg:
        #                neg = self.negative[random.randint(0, len(self.negative) - 1)] 
        #            label[i,j,k] = neg 
        #label = mx.nd.array(label, dtype='int32')

        # [bptt,batch_size]
        tmp1 = self._source[i+1:i+1+self._bptt].astype(np.int32)
        tmp1 = tmp1.expand_dims(0)

        tmp2 = self.negative_sample((self.num_label-1, self._bptt, self.batch_size))
        tmp2 = mx.nd.array(tmp2, dtype='int32')

        label = mx.nd.concat(tmp1, tmp2, dim=0)
        label = label.transpose((1,2,0))

        # generate weight
        label_weight = mx.nd.zeros(label_shape, dtype='float32')
        label_weight[0,:,:] = 1.0
        label_weight = label_weight.transpose((1,2,0))

        self._next_label = (label, label_weight)
        return True

    def negative_sample(self, shape):
        sz = 1
        for v in shape:
            sz *= v

        res = np.zeros((sz,))
        for i in range(0,sz):
            res[i] = self.negative[random.randint(0, len(self.negative) - 1)] 

        return res.reshape(shape)

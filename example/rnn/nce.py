#!/usr/bin/env python
# coding: utf-8
#
# Usage: 
# Author: weixing.mei(auimoviki@gmail.com)

from __future__ import print_function

import bisect
import random
import numpy as np

import mxnet as mx
from mxnet import ndarray
from mxnet.metric import EvalMetric
from mxnet.io import DataIter, DataBatch

class NceMetric(EvalMetric):
    def __init__(self, ignore_label, axis=-1):
        super(NceMetric, self).__init__('NCE')
        self.ignore_label = ignore_label
        self.axis = axis

    def update(self, labels, preds):
        """
        compute nce loss
        """
        labwgt = [labels[1]]
        labels = [labels[0]]
        assert len(labels) == len(preds) == len(labwgt)

        # batch_size, seq_len, num_label
        shape = labels[0].shape
        num_label = shape[-1]

        loss = 0.
        num = 0
        probs = []

        for pred,lab in zip(preds, labels):
            lab = lab.reshape(shape=(-1, num_label)).as_in_context(pred.context)
            flag = 1-(lab==self.ignore_label)
            loss += -ndarray.sum(pred*flag).asnumpy()[0]

        loss /= (shape[1]*shape[2])

        self.sum_metric += loss
        self.num_inst += 1



def nce_loss(data, label, label_weight, embed_weight, vocab_size, num_hidden, num_label, seq_len):
    """
    data format: NT

    PARAMETERS:
        - data: input data, lstm layer output, size [batch_size, seq_len, num_hidden]
        - label: input label, size: [seq_len, num_label, batch_size], the first one is true label
        - label_weight: weight of each label, [seq_len, num_label, batch],
          first is 1.0, others are 0.0
        - embed_weight: embeding matrix for label embeding 
        - vocab_size: the size of vocab
        - num_hidden: length of hidden
        - num_label: length of label
    """
    # [batch_size, seq_len, num_label] ->  [batch_size, seq_len, num_label, num_hidden]
    label_embed = mx.sym.Embedding(data = label, weight=embed_weight, 
                                   input_dim = vocab_size,
                                   output_dim = num_hidden, name = 'output_embed')

    # data: [batch_size, seq_len, num_hidden] 
    # label_embed: [batch_size, seq_len, num_label, num_hidden]
    #
    # output: [batch_size, seq_len, num_label, num_hidden]
    data = mx.sym.Reshape(data=data, shape=(-1, 1, num_hidden))
    label_embed = mx.sym.Reshape(data=label_embed, shape=(-1, num_label, num_hidden))
    pred = mx.sym.broadcast_mul(data, label_embed)

    # [batch_size, seq_len, num_label]
    pred = mx.sym.sum(data=pred, axis=2)

    label_weight = mx.sym.Reshape(data=label_weight, shape=(-1, num_label))

    # pred: [seq_len, batch_sie, num_label]
    # label_weight: [batch_size, seq_len, num_label]
    # output: [batch_size, seq_len, num_label]
    return mx.sym.LogisticRegressionOutput(data = pred, label = label_weight)


class LMNceIter(DataIter):
    """
    Simple bucketing iterator for nce language model.

    Label for each step is constructed from data of next step and random sample 

    PARAMETERS
    ----------
    - sentences : list of list of int 
        encoded sentences

    - batch_size : int 
        batch_size of data

    - invalid_label : int 
        default -1,  key for invalid label, e.g. <end-of-sentence>

    - dtype : str, default 'float32'
        data type
        
    - buckets : list of int
        size of data buckets. Automatically generated if None.

    - data_name : str, default 'data'
        name of data

    - label_name : str, default 'label'
        name of label

    - layout : str
        format of data and label. 'NT' means (batch_size, length)
        and 'TN' means (length, batch_size).

    - freq: list of int
        frequence of each word
    """
    def __init__(self, sentences, batch_size, freq, buckets=None, invalid_label=-1,
                 data_name='data', label_name='label', dtype='float32',
                 layout='NTC', num_label=5):
        super(DataIter, self).__init__()

        # generate buckets automatically
        if not buckets:
            buckets = [i for i, j in enumerate(np.bincount([len(s) for s in sentences]))
                       if j >= batch_size]
        buckets.sort()
        self.buckets = buckets

        # put sentences into each bucket
        ndiscard = 0
        self.data = [[] for _ in buckets]
        for i, sent in enumerate(sentences):
            buck = bisect.bisect_left(buckets, len(sent))
            # throw sentences that length bigger than the largest bucket length
            if buck == len(buckets):
                ndiscard += 1
                continue
            buff = np.full((buckets[buck],), invalid_label, dtype='int64')
            buff[:len(sent)] = sent
            self.data[buck].append(buff)

        print("WARNING: discarded %d sentences longer than the largest bucket."%ndiscard)

        # list of list(sentence)
        self.data = [np.asarray(i, dtype=dtype) for i in self.data]

        self.freq = freq

        self.dtype = dtype

        self.data_name = data_name
        self.label_name = label_name
        self.label_weight_name = "%s_weight" % label_name 

        self.num_label = num_label
        self.invalid_label = invalid_label

        self.nddata = []
        self.ndlabel = []
        self.ndlabel_weight = []

        self.batch_size = batch_size
        self.default_bucket_key = max(buckets)

        self.major_axis = layout.find('N')

        if self.major_axis == 0:
            self.provide_data = [(data_name, (batch_size, self.default_bucket_key))]
            self.provide_label = [
                    (self.label_name, (batch_size, self.default_bucket_key, self.num_label)),
                    (self.label_weight_name, (batch_size, self.default_bucket_key, self.num_label))
                    ]
        elif self.major_axis == 1:
            self.provide_data = [(data_name, (self.default_bucket_key, batch_size))]
            self.provide_label = [
                    (self.label_name, (self.default_bucket_key, batch_size, self.num_label)),
                    (self.label_weight_name, (self.default_bucket_key, batch_size, self.num_label))
                    ]
        else:
            raise ValueError("Invalid layout %s: Must by NT (batch major) or TN (time major)")

        # self.idx[0] = (i,j)
        #   i: the i'th bucket, whose length is self.buckets[i]
        #   j: the start index of batch in self.data[i] 
        #   
        # get a batch of data, self.data[0][j:j+batch_size]
        self.idx = []
        for i, buck in enumerate(self.data):
            self.idx.extend([(i, j) for j in range(0, len(buck) - batch_size + 1, batch_size)])
        self.curr_idx = 0

        self.reset()

    def reset(self):
        print("reset")
        self.curr_idx = 0
        # shuffle index
        random.shuffle(self.idx)

        # shuffle sentences in a bucket  
        for buck in self.data:
            np.random.shuffle(buck)

        # buffer for negtive sample
        negbuf = [int(np.power(i, 0.75)) for i in self.freq]
        negbuf = []
        for i in self.freq:
            val = int(np.power(i, 0.75))
            negbuf.extend(np.full(val, i))
        random.shuffle(negbuf)

        self.nddata = []
        self.ndlabel = []
        self.ndlabel_weight = []

        negLen = len(negbuf) 

        for buck in self.data:
            # buck is a list of list(sentences), each row stand for a sentences
            #
            # each sentence should generate a matrix represent labels(totally num_label column)
            #
            # size of label:
            #   0: batch_size 
            #   1: sentences length 
            #   2: num_label 
            buckLab = []
            buckLabWgt = []

            for sent in buck:
                shape = (len(sent), self.num_label)

                wgt = np.zeros(shape)
                wgt[:, 0] = 1
                buckLabWgt.append(wgt)

                label = np.full(shape, self.invalid_label)
                label[:-1,0] = sent[1:]

                for i,wrd in enumerate(sent):
                    # negative sample
                    j = 1
                    while j<self.num_label:
                        val = np.random.randint(negLen)
                        val = negbuf[val]
                        if val!=wrd:
                            label[i][j] = val
                            j+= 1
                buckLab.append(label)

	    # data format: NT
	    # label format: NTL
	    # N: batch number
	    # T: time stamp
	    # L: label
            self.nddata.append(ndarray.array(buck, dtype=self.dtype))
            self.ndlabel.append(ndarray.array(buckLab, dtype=self.dtype))
            self.ndlabel_weight.append(ndarray.array(buckLabWgt, dtype=self.dtype))

    def next(self):
        if self.curr_idx == len(self.idx):
            raise StopIteration
        i, j = self.idx[self.curr_idx]
        self.curr_idx += 1
        # print('sample iter: ', i, j)

	step = self.batch_size
        if self.major_axis == 1:
            data = self.nddata[i][j:j+step].T

	    #NTL -> TNL
            label = ndarray.transpose(data=self.ndlabel[i][j:j+step], axes=(1,0,2))
            label_weight = ndarray.transpose(data=self.ndlabel_weight[i][j:j+step], axes=(1,0,2))
        else:
            data = self.nddata[i][j:j+step]
            label = self.ndlabel[i][j:j+step]
            label_weight = self.ndlabel_weight[i][j:j+step]

        return DataBatch([data], [label, label_weight], pad=0,
                         bucket_key=self.buckets[i],
                         provide_data=[(self.data_name, data.shape)],
                         provide_label=[
                             (self.label_name, label.shape),
                             (self.label_weight_name, label_weight.shape)
                             ])

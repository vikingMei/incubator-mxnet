#!/usr/bin/env python
# coding: utf-8
#
# Usage: 
# Author: weixing.mei(auimoviki@gmail.com)

from __future__ import print_function

import bisect
import numpy as np
import threading
import multiprocessing as mp

import mxnet as mx
from mxnet import ndarray
from mxnet.metric import EvalMetric
from mxnet.io import DataIter, DataBatch

#np.random.seed(0)

@mx.init.register
class MyConstant(mx.init.Initializer):
    def __init__(self, value):
        super(MyConstant, self).__init__(value=value)
        self.value = value

    def _init_weight(self, _, arr):
        arr[:] = mx.nd.array(self.value)


class NceMetric(EvalMetric):
    def __init__(self, ignore_label, axis=-1):
        super(NceMetric, self).__init__('NCE')
        self.ignore_label = ignore_label
        self.axis = axis

    def get(self):
        if self.num_inst == 0:
            return (self.name, float('nan'))
        else:
            return (self.name, self.sum_metric / self.num_inst)


    def update(self, labels, preds):
        """
        compute nce loss
        """
        # remove other internals output symbols
        preds = [preds[0]]

        assert len(labels) == 2*len(preds)

        labvals = [labels[0]]
        labwgts = [labels[1]]

        # batch_size, seq_len, num_label
        shape = labels[0].shape
        num_label = shape[-1]

        loss = 0.
        num = 0
        probs = []

        for pred,labval,labwgt in zip(preds, labvals, labwgts):
            #pred.asnumpy().tofile('./logs/%05d', self.idx)
            labval = labval.as_in_context(pred.context)
            labval = labval.reshape(pred.shape)
            labwgt = labwgt.as_in_context(pred.context)
            labwgt = labwgt.reshape(pred.shape)

            # p*log(q) + (1-p)*log(1-q)
            pred = labwgt*ndarray.log(ndarray.maximum(1e-10, pred)) + (1-labwgt)*ndarray.log(ndarray.maximum(1e-10, 1-pred))

            # mask invalid label
            if self.ignore_label is not None:
                flag = (labval==self.ignore_label)
                pred = pred*(1-flag)
                num -= ndarray.sum(flag).asscalar()

            num += pred.size
            loss -= ndarray.sum(pred).asscalar()

        self.sum_metric += loss
        self.num_inst += num/num_label


def nce_loss(data, label, label_weight, vocab_size, num_hidden, num_label, seq_len, pad_label):
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
    # [batch_size*seq_len, 1, num_hidden]
    data = mx.sym.Reshape(data, shape=(-1, 1, num_hidden), name='pred_reshape')

    # [batch_size, seq_len, num_label] ->  [batch_size, seq_len, num_label, num_hidden]
    labemb_wgt = mx.sym.Variable('label_embed_weight')
    label_embed = mx.sym.Embedding(label, input_dim = vocab_size,
                                   output_dim = num_hidden, weight=labemb_wgt, name = 'label_embed')

    # [batch_size*seq_len, num_label, num_hidden]
    label_embed = mx.sym.Reshape(label_embed, shape=(-1, num_label, num_hidden), name='label_embed_reshape')

    # [batch_size*seq_len, num_label, 1]
    bias = mx.sym.Embedding(label, input_dim=vocab_size, output_dim=1, name="bias_embed")
    bias = mx.sym.Reshape(bias, shape=(-1, num_label), name='bias_embed_reshape')

    # [batch_size*seq_len, num_label, num_hidden]
    pred = mx.sym.broadcast_mul(data, label_embed, name='pred_labemb_broadcast_mul')

    # [batch_size*seq_len, num_label]
    pred = mx.sym.sum(data=pred, axis=2, name='pred_labemb_broadcast_mul_sum')

    pred = pred + bias

    # mask out pad data
    pad_label = mx.sym.Variable('pad_label', shape=(1,), init=MyConstant([pad_label]))
    label = mx.sym.Reshape(data=label, shape=(-1,num_label), name='label_reshape')
    flag = mx.sym.broadcast_not_equal(lhs=label, rhs=pad_label, name='mask_gen')

    pred = pred*flag 

    label_weight = mx.sym.Reshape(data=label_weight, shape=(-1, num_label), name='labwgt_reshape')
    pred =  mx.sym.LogisticRegressionOutput(data=pred, label=label_weight, name='final_logistic')

    return pred



class dataPrepareProcess(mp.Process):
    """
    sub-process used to papare data
    """
    def __init__(self, name, data, negbuf, num_label, pad_label, outq):
        mp.Process.__init__(self)

        self.name = name
        self.data = data
        self.negbuf= negbuf
        self.num_label = num_label
        self.pad_label = pad_label
        self.outq = outq

    def run(self):
        output = { 'label': [], 'weight': [] }

        neglen = len(self.negbuf)
        sentcnt = 0
        for sent in self.data:
            sentcnt += 1
            shape = (len(sent), self.num_label)

            labwgt = np.zeros(shape)
            labwgt[:, 0] = 1

            label = np.full(shape, self.pad_label)
            label[:-1,0] = sent[1:]

            for i,wrd in enumerate(sent):
                truelabel = label[i][0]
                if truelabel==self.pad_label:
                    labwgt[i:,] = 0.5
                    break

                # unique negative sample
                valset = {truelabel: 1}
                cnt = 0
                while len(valset)<self.num_label:
                    cnt += 1
                    val = np.random.randint(neglen)
                    val = self.negbuf[val]
                    if val!=wrd:
                        valset[val] = 1
                valset.pop(truelabel)
                label[i, 1:] = valset.keys()

            output['label'].append(label)
            output['weight'].append(labwgt)

        output['data'] = self.data
        self.outq.put_nowait(output)


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

    - pad_label : int 
        default -1,  key for invalid label, used for pending

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
    def __init__(self, sentences, batch_size, freq, buckets=None, pad_label=-1,
                 data_name='data', label_name='label', dtype='float32',
                 layout='NTC', num_label=5, for_train=True):
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
            buff = np.full((buckets[buck],), pad_label, dtype='int64')
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
        self.pad_label = pad_label

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

        self.for_train = for_train
        if for_train:
            self.reset()
        else:
            self.cur_idx = 0
            np.random.shuffle(self.idx)
            self.prepare()


    def reset(self):
        if self.for_train:
            self.curr_idx = 0

            # shuffle index
            np.random.shuffle(self.idx)
            self.prepare()

    def prepare(self):
        # shuffle sentences in a bucket  
        for buck in self.data:
            np.random.shuffle(buck)

        # buffer for negtive sample
        negbuf = []
        for i,cnt in self.freq.items():
            cnt = int(np.power(cnt, 0))
            negbuf.extend(np.full(cnt, i))
        np.random.shuffle(negbuf)

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
            buckData = []
            buckLabWgt = []

            numProc = min(mp.cpu_count(), len(buck))

            if numProc>0:
                procPoll = []
                procStep = int(len(buck)/numProc)

                posData = 0
                outq = mp.Queue(maxsize=numProc)
                for i in range(0, numProc-1):
                    ins = dataPrepareProcess('%03d' %i, buck[posData:posData+procStep], negbuf, self.num_label, self.pad_label, outq)
                    ins.start()
                    procPoll.append(ins)
                    posData += procStep

                ins = dataPrepareProcess('%03d' % numProc, buck[posData:], negbuf, self.num_label, self.pad_label, outq)
                ins.start()
                procPoll.append(ins)

                for ins in procPoll:
                    res = outq.get()
                    if res:
                        buckLab.extend(res['label'])
                        buckData.extend(res['data'])
                        buckLabWgt.extend(res['weight'])

                for ins in procPoll:
                    ins.join()

	    # data format: NT
	    # label format: NTL
	    # N: batch number
	    # T: time stamp
	    # L: label
            self.nddata.append(ndarray.array(buckData, dtype=self.dtype))
            self.ndlabel.append(ndarray.array(buckLab, dtype=self.dtype))
            self.ndlabel_weight.append(ndarray.array(buckLabWgt, dtype=self.dtype))


    def next(self, i=None, j=None):
        if self.curr_idx == len(self.idx):
            raise StopIteration

        if i is None or j is None:
            i, j = self.idx[self.curr_idx]
            #i, j = self.idx[0]
            self.curr_idx += 1

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

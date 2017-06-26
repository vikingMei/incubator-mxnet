#!/usr/bin/env python
# coding: utf-8
#
# Usage: 
# Author: Summer Qing(qingyun.wu@aispeech.com)

import numpy as np
from mxnet import ndarray
from mxnet.metric import EvalMetric

class NceMetric(EvalMetric):
    def __init__(self, num_lab, negdis, ignore_label=0, axis=-1, step=1):
        super(NceMetric, self).__init__('NCE')
        self.axis = axis
        self.idx = 0
        self.step = step

        self.negdis = negdis
        if type(negdis)!=ndarray.NDArray:
            self.negdis = ndarray.array(negdis).reshape((-1,1))
        else:
            self.negdis = negdis.reshape((-1, 1))

        self.num_label = num_lab
        self.ignore_label = ignore_label

        k = num_lab-1
        self.klogk =  k*np.log(k)

        self.vocab_size = self.negdis.size


    def get(self):
        if self.num_inst == 0:
            return (self.name, float('nan'))
        else:
            return (self.name, self.sum_metric / self.num_inst)


    def update(self, labels, preds):
        """
        compute nce loss
        """
        assert len(labels) == 2*len(preds)

        labvals = [labels[0]]
        labwgts = [labels[1]]

        k = self.num_label-1

        loss = 0.
        num = 0

        # ml(theta, w) = k*log(k) + log(y_0) + sum_neg(log(pn_w)) - sum(log(y+k*pn_w)
        #
        # loss = -ml(theta, w) = sum(log(y+k*pn) - log(y_0) - sum_neg(log(pn)) - k*log(k)
        for pred,label,labwgt in zip(preds, labvals, labwgts):
            #[batch_size, seq_len, num_label]
            label = label.as_in_context(pred.context)
            label = label.reshape(pred.shape)

            negdis = self.negdis.as_in_context(pred.context)

            # [batch_size, seq_len, num_label]
            pn = ndarray.Embedding(label, weight=negdis, input_dim=self.vocab_size, output_dim=1) 
            pn = pn.reshape(label.shape).as_in_context(pred.context)

            pred = ndarray.maximum(1e-10, pred)
            pn = ndarray.maximum(1e-10, pn)

            acc = ndarray.log(pred+k*pn) 

            # ndarray not support multiple dimension slice, so using mask for help
            mask = np.zeros(acc.shape) 
            mask[:, :, 0] = 1.0
            mask = ndarray.array(mask, ctx=pred.context)

            # -log(y_0) 
            # acc[:, :, 0] -= ndarray.log(pred[:, :, 0]) 
            acc -= ndarray.log(pred)*mask

            #acc[:, :, 1:] -= ndarray.log(pn[:, :, 1:]) 
            acc -= ndarray.log(pn)*(1-mask)

            # mask invalid label
            if self.ignore_label is not None:
                flag = (label==self.ignore_label)
                acc = acc*(1-flag)
                num -= ndarray.sum(flag).asscalar()

            num += pred.size
            #self.idx += 1
            #if 1==self.idx%self.step:
            #    fname = 'output/logs/%03d-acc' % (self.idx)
            #    print(fname)
            #    acc.asnumpy().tofile(fname, sep='\n')

            loss += ndarray.sum(acc).asscalar() 
            loss += self.klogk

        self.sum_metric += loss
        self.num_inst += num/self.num_label

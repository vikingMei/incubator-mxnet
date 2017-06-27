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
            label = label.reshape(pred.shape)

            # [batch_size, seq_len, num_label]
            pn = ndarray.Embedding(label, weight=self.negdis, input_dim=self.vocab_size, output_dim=1) 
            pn = pn.reshape(label.shape)

            pred = pred.asnumpy()
            pn  = pn.asnumpy()

            pospred = pred[:, :, 0]
            negpred = pred[:, :, 1:]

            pospn = pn[:, :, 0]
            negpn = pn[:, :, 1:]

            posloss = np.log(pospred/(pospred+k*pospn))

            knegpn = k*negpn
            negloss = np.log(np.maximum(1e-10, knegpn)/(negpred+knegpn))

            # mask invalid label
            if self.ignore_label is not None:
                flag = (label==self.ignore_label).asnumpy()
                num -= np.sum(flag)

                flag = 1-flag
                posloss = posloss*flag[:, :, 0] 
                negloss = negloss*flag[:, :, 1:] 

            num += pred.size
            loss -= (negloss.sum()+posloss.sum())

        self.sum_metric += loss
        self.num_inst += num/self.num_label

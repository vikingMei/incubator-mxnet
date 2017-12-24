#!/usr/bin/env python
# coding: utf-8
#
# Usage: 

import time
import logging
import numpy as np
import mxnet as mx
from mxnet import ndarray
from mxnet.metric import EvalMetric

class NceMetric(EvalMetric):
    def __init__(self, num_lab, negdis, ignore_label=0, axis=-1, step=1):
        super(NceMetric, self).__init__('NCE')
        self.axis = axis
        self.idx = 0
        self.step = step
        self.ignore_label = ignore_label
        self.numlab = num_lab

    def get(self):
        if self.num_inst == 0:
            #logging.warn('num of instance is [0] in %s' % self.name)
            res = (self.name, float('nan'))
        else:
            res = (self.name, self.sum_metric/self.num_inst)
        return res


    def update(self, labels, preds):
        """
        compute nce loss
        """
        assert len(labels) == 2*len(preds)

        labvals = [labels[0]]
        labwgts = [labels[1]]

        loss = 0.
        num = 1

        for pred,label,labwgt in zip(preds, labvals, labwgts):
            idx = 0
            ## mask invalid label
            if self.ignore_label is not None:
                num -= ndarray.sum(1-labwgt).asscalar()

            tmp = ndarray.log(ndarray.maximum(pred, 1.0e-15))

            tmp = tmp.reshape((-1,)).as_in_context(labwgt.context)
            labwgt = labwgt.reshape((-1,))
            tmp = ndarray.sum(tmp*labwgt)

            loss -= tmp.asscalar()
            num += pred.size

        self.num_inst += num/self.numlab
        self.sum_metric += loss

#!/usr/bin/env python
# coding: utf-8
#
# Usage: 

import os
import mxnet as mx
import numpy as np
import logging


class NceOutput(mx.operator.CustomOp):
    def __init__(self):
        super(NceOutput, self).__init__()
        self.idx = 0
        
    def forward(self, is_train, req, in_data, out_data, aux):
        """
        PARAMETER:
            - is_train: bool, is train or not
            - req:      define how to save data to out_data 
            - in_data:  input data, in_data[0] is data, in_data[1] is label
            - out_data: used to save forward result
            - aux:      auxiliary data, not used in this operator 
        """
        data = in_data[0]
        label = in_data[1]
        lnz = in_data[4]

        y = mx.nd.exp(data-lnz)

        self.assign(out_data[0], req[0], y)


    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        # [batch_size, seq_len]
        data = in_data[0]

        # [batch_size, seq_len, num_lab]
        label= in_data[1]
        labwgt=in_data[2]

        numlab = label.shape[-1]
        k = numlab-1

        # [batch_size, seq_len, num_lab]
        pn = in_data[3]

        # [batch_size, seq_len, num_lab]
        y = out_data[0]

        # [batch_size, seq_len, num_lab]
        label = label.reshape((-1,))

        grad = -y/(y+k*pn)

        # mask 
        mask = np.zeros(grad.shape)
        mask[:, :, 0] = 1
        mask = mx.nd.array(mask).as_in_context(grad.context)

        grad = grad+mask
        grad = -grad*labwgt

        self.idx += 1
        #fname = './output/gradient/%03d' % self.idx
        #print(fname)
        #grad.asnumpy().tofile(fname, sep='\n')

        self.assign(in_grad[0], req[0], grad)


@mx.operator.register("NceOutput")
class NceOutputProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(NceOutputProp, self).__init__(need_top_grad=False)
    
    def list_arguments(self):
        return ['data', 'label', 'label_weight', 'negprob', 'lnz']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        output_shape = in_shape[0]
        return in_shape, [output_shape], []

    def infer_type(self, in_type):
        return in_type, [in_type[0]], []

    def create_operator(self, ctx, shapes, dtypes):
        return NceOutput()

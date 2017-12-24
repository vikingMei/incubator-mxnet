#!/usr/bin/env python
# coding: utf-8
#
# Usage: 

import os
import mxnet as mx
import numpy as np
import logging


class MyLogistic(mx.operator.CustomOp):
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

        y = 1+mx.nd.exp(-data) 
        y = 1/y

        print('VIKING forward data: ', data.asnumpy())
        print('VIKING forward y: ', y.asnumpy())
        #x = in_data[0].asnumpy()
        #y = np.exp(x - x.max(axis=1).reshape((x.shape[0], 1)))
        #y /= y.sum(axis=1).reshape((x.shape[0], 1))
        self.assign(out_data[0], req[0], y)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        data = in_data[0]
        label= in_data[1]
        y = data - label

        print('VIKING backward data: ', data.asnumpy())
        print('VIKING backward lab: ', label.asnumpy())
        print('VIKING backward y: ', y.asnumpy())

        self.assign(in_grad[0], req[0], y)
        self.assign(in_grad[1], req[1], y)

@mx.operator.register("MyLogistic")
class MyLogisticProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(MyLogisticProp, self).__init__(need_top_grad=False)
    
    def list_arguments(self):
        return ['data', 'label']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        label_shape = in_shape[1]
        output_shape = in_shape[0]
        return [data_shape, label_shape], [output_shape], []

    def infer_type(self, in_type):
        return in_type, [in_type[0]], []

    def create_operator(self, ctx, shapes, dtypes):
        return MyLogistic()

#!/usr/bin/env python
# coding: utf-8
#
# Usage: 
# Author: wxm71(weixing.mei@aispeech.com)

import sys
import numpy as np
import mxnet as mx

a = mx.ndarray.ones((1,))
b = mx.ndarray.zeros((1,))

mx.ndarray.LogisticRegressionOutput(a, b)
# LogisticRegressionOutput(1, 0):  [ 0.7310586]
# LogisticRegressionOutput(0, 1):  [ 0.5]
print 'LogisticRegressionOutput(1, 0): ', mx.ndarray.LogisticRegressionOutput(a, b).asnumpy()
print 'LogisticRegressionOutput(0, 1): ', mx.ndarray.LogisticRegressionOutput(b, a).asnumpy()

# 1/[1+exp(-x)]
def myfunc(h, p):
    q = 1/(1+np.exp(-h))

    #p log(q) + (1-p)*log(1-q)
    return q
    #return  p*np.log(q) + (1-p)*np.log(1-q)

print myfunc(1,0),  myfunc(0, 1)

vara = mx.sym.Variable('a')
varb = mx.sym.Variable('b')
pred = mx.sym.Activation(data=varb, act_type='sigmoid')
exec_ =  pred.bind(ctx=mx.cpu(), args={'b':b})
exec_.forward()
print 'mx.sym.LogisticRegressionOutput(1, 0): ', exec_.outputs[0].asnumpy()
# 
# pred = mx.sym.LogisticRegressionOutput(data=varb, label=vara)
# exec_ =  pred.bind(ctx=mx.cpu(), args={'a':a, 'b':b})
# exec_.forward()
# print 'mx.sym.LogisticRegressionOutput(0, 1): ', exec_.outputs[0].asnumpy()

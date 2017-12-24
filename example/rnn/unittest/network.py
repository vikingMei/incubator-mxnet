#!/usr/bin/env python 
# coding: utf-8
#
# Usage: 

import sys
import mxnet as mx
from nce import NceMetric
from lstm_nce import train_sym_gen
from toy_data import gen_toy_data, gen_train_data


if "__main__"==__name__:
    reload(sys)
    sys.setdefaultencoding('utf-8')

    class args:
        num_layers = 2 
        num_hidden = 200 
        num_embed = 200 
        num_label = 5
        batch_size = 1
        dropout = 0

    dataIter = gen_toy_data()

    cell, sym_gen = train_sym_gen(args, len(dataIter.vocab), dataIter.pad_label)

    net,data_name,label_name = sym_gen(5)
    data = dataIter.next()

    print dataIter.raw_data
    print '\ndata: \n', data.data[0].asnumpy()
    print '\nlabel: \n', data.label[0].asnumpy()
    print '\nlabel weight: \n', data.label[1].asnumpy()

    mod = mx.mod.Module(symbol=net, context=mx.gpu(0), data_names=data_name, label_names=label_name)
    mod.bind(data_shapes=dataIter.provide_data, label_shapes=dataIter.provide_label)
    mod.init_params(initializer=mx.init.Uniform(scale=.1))
    mod.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate', 0.1), ))
    mod.forward(data)
    
    pred = mod.get_outputs()
    print '\n'
    print 'output: ', pred[0].asnumpy()

    metric = NceMetric(dataIter.pad_label)
    mod.update_metric(metric, data.label)
    print metric.sum_metric
    print metric.num_inst

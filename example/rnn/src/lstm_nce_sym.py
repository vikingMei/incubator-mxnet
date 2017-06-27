#!/usr/bin/env python
# coding: utf-8
#
# Usage: 
# Author: Summer Qing <qingyun.wu@aispeech.com>

import sys

import numpy as np
import mxnet as mx

from lstm_nce import NceOutput



@mx.init.register
class MyConstant(mx.init.Initializer):
    def __init__(self, value):
        typ=type(value)

        if mx.nd.NDArray==typ:
            value = value.asnumpy().tolist()

        if type(value)==np.ndarray:
            value = value.tolist()

        super(MyConstant, self).__init__(value=value)

        self.value = value

    def _init_weight(self, _, arr):
        arr[:] = mx.nd.array(self.value)



def train_sym_gen(args, vocab_size, pad_label, negdisval=None):
    cell = mx.rnn.FusedRNNCell(args.num_hidden, num_layers=args.num_layers, dropout=args.dropout, mode='lstm')
    #cell = mx.rnn.SequentialRNNCell()
    #for i in range(args.num_layers):
    #    cell.add(mx.rnn.LSTMCell(num_hidden=args.num_hidden, prefix='lstm_l%d_'%i))

    def sym_gen(seq_len):
        # [batch_size, seq_len]
        data = mx.sym.Variable('data')
        label = mx.sym.Variable('label')
        labwgt = mx.sym.Variable('label_weight')

        negdis = mx.sym.Variable('negdis', shape=(vocab_size,1), dtype='float32', init=MyConstant(negdisval.reshape(-1,1)) )
        negdis = mx.sym.BlockGrad(negdis)

        lnz = mx.sym.Variable('lnz', shape=(1), dtype='float32', init=mx.init.Constant(args.lnz))
        lnz = mx.sym.BlockGrad(lnz)

        # map input to a embeding vector
        embedIn = mx.sym.Embedding(data, input_dim=vocab_size, output_dim=args.num_embed,name='input_embed')

        # pass embedding vector to lstm
        # [batch_size, seq_len, num_hidden]
        pred, _ = cell.unroll(seq_len, inputs=embedIn, layout='NTC', merge_outputs=True)

        #[batch_size, seq_len, num_lab, num_hidden]
        labemb = mx.sym.Embedding(label, input_dim=vocab_size, output_dim=args.num_embed, name="output_embed")

        # [batch_size, seq_len, num_lab, 1]
        biaswgt = mx.sym.Variable('bias_embed_weight',init=mx.init.Zero())
        biasemb = mx.sym.Embedding(label, input_dim=vocab_size, output_dim=1, weight=biaswgt, name="bias_embed")
        biasemb = mx.sym.Reshape(biasemb, shape=(-1, 0, 0))

        # [batch_size, seq_len, 1, num_hidden]
        pred = mx.sym.expand_dims(pred, axis=2, name='pred_expand_dim')

        # [batch_size, seq_len, num_lab]
        pred = mx.sym.broadcast_mul(pred, labemb, name="broadcast_mul")
        pred = mx.sym.sum(pred, axis=3, name="broadcast_mul_sum")

        pred = pred + biasemb

        # find Pn[lab] using embeding
        negprob = mx.sym.Embedding(label,input_dim=vocab_size, output_dim=1, weight=negdis, name="negprob_embeding")
        negprob = mx.sym.Reshape(negprob, shape=(-1, 0, args.num_label))
        negprob = mx.sym.BlockGrad(negprob, name='negprob_stop_gradient')

        pred = mx.symbol.Custom(data=pred, label=label, label_weight=labwgt,
                    negprob=negprob, lnz=lnz, 
                    name='nce_output', op_type='NceOutput')

        return pred, ('data',), ('label', 'label_weight')

    return cell, sym_gen


def test_sym_gen(args, vocab_size): 
    stack = mx.rnn.FusedRNNCell(args.num_hidden, num_layers=args.num_layers, mode='lstm')

    def sym_gen(seq_len):
        data = mx.sym.Variable('data')
        embed = mx.sym.Embedding(data=data, input_dim=vocab_size, output_dim=args.num_embed, name='input_embed')

        # [batch_size, seq_len, num_hidden]
        pred, states = stack.unroll(seq_len, inputs=embed, layout='NTC', merge_outputs=True)

        # get output embedding
        # TODO: this is a constant, initialize it only one-time 
        #
        # [vocab_size] -> [vocab_size, num_hidden]
        allLab = mx.sym.Variable('alllab', shape=(vocab_size,), dtype='float32')
        labs = mx.sym.Embedding(data=allLab, input_dim=vocab_size, output_dim=args.num_hidden, name='output_embed')

        # [vocab_size]
        bias = mx.sym.Embedding(allLab, input_dim=vocab_size, output_dim=1, name='bias_embed')
        bias = mx.sym.Reshape(bias, shape=(-1,))

        # [batch_size*seq_len, vocab_size, num_hidden]
        pred = mx.sym.Reshape(pred, shape=(-1, 1, args.num_hidden))
        pred = mx.sym.broadcast_mul(pred, labs)

        # [batch_size*seq_len, vocab_size]
        pred = mx.sym.sum(data=pred, axis=2)
        pred = mx.sym.broadcast_add(pred, bias)

        label = mx.sym.Variable('label')
        label = mx.sym.Reshape(label, shape=(-1,))
        pred = mx.sym.SoftmaxOutput(data=pred, label=label, name='softmax')

        return pred, ('data',), ('label',)

    return sym_gen, stack

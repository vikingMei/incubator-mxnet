#!/usr/bin/env python
# coding: utf-8
#
# Usage: 
# Author: wxm71(weixing.mei@aispeech.com)

import sys

import numpy as np
import mxnet as mx

from lstm_nce import NceOutput


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

        # map input to a embeding vector
        embedIn = mx.sym.Embedding(data, input_dim=vocab_size, output_dim=args.num_embed,name='input_embed')

        # pass embedding vector to lstm
        # [batch_size, seq_len, num_hidden]
        pred, _ = cell.unroll(seq_len, inputs=embedIn, layout='TNC', merge_outputs=True)

        #[batch_size, seq_len, num_lab, num_hidden]
        labemb = mx.sym.Embedding(label, input_dim=vocab_size, output_dim=args.num_embed, name="output_embed")

        # [batch_size, seq_len, num_lab, 1]
        biaswgt = mx.sym.Variable('bias_embed_weight',init=mx.init.Zero())
        biasemb = mx.sym.Embedding(label, input_dim=vocab_size, output_dim=1, weight=biaswgt, name="bias_embed")
        biasemb = mx.sym.Reshape(biasemb, shape=(-1, seq_len, args.num_label))

        # [batch_size, seq_len, 1, num_hidden]
        pred = mx.sym.Reshape(pred, shape=(-1, seq_len, 1, args.num_hidden), name="pred_reshape")

        # [batch_size, seq_len, num_lab]
        pred = mx.sym.broadcast_mul(pred, labemb, name="broadcast_mul")
        pred = mx.sym.sum(pred, axis=3, name="broadcast_mul_sum")

        pred = pred + biasemb

        # find Pn[lab] using embeding
        negprob = mx.sym.Embedding(label,input_dim=vocab_size, output_dim=1, weight=negdis, name="negprob_embeding")
        negprob = mx.sym.Reshape(negprob, shape=(-1,seq_len, args.num_label))
        negprob = mx.sym.BlockGrad(negprob, name='negprob_stop_gradient')

        pred = mx.symbol.Custom(data=pred, label=label, label_weight=labwgt, negprob=negprob, name='nce_output', op_type='NceOutput')

        return pred, ('data',), ('label', 'label_weight')

    return cell, sym_gen


def test_sym_gen(args, vocab_size): 
    if not args.stack_rnn:
        stack = mx.rnn.FusedRNNCell(args.num_hidden, num_layers=args.num_layers,
                mode='lstm', bidirectional=args.bidirectional).unfuse()
    else:
        stack = mx.rnn.SequentialRNNCell()
        for i in range(args.num_layers):
            cell = mx.rnn.LSTMCell(num_hidden=args.num_hidden, prefix='lstm_l%d_'%i)
            stack.add(cell)

    def sym_gen(seq_len):
        data = mx.sym.Variable('data')
        embed = mx.sym.Embedding(data=data, input_dim=vocab_size, output_dim=args.num_embed, name='input_embed')

        stack.reset()

        # [batch_size, seq_len, num_hidden]
        pred, states = stack.unroll(seq_len, inputs=embed, layout='NTC', merge_outputs=True)

        # get output embedding
        # TODO: this is a constant, initialize it only one-time 
        #
        # [vocab_size] -> [vocab_size, num_hidden]
        allLab = mx.sym.Variable('alllab', shape=(vocab_size-1,), dtype='float32')
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

        pred = mx.sym.sigmoid(data=pred, name='sigmoid')

        label = mx.sym.Variable('label')
        label = mx.sym.Reshape(label, shape=(-1,))
        pred = mx.sym.SoftmaxOutput(data=pred, label=label, name='softmax')

        return pred, ('data',), ('label',)

    return sym_gen, stack

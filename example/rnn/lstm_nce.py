#!/usr/bin/env python
# coding: utf-8
#
# Usage: 
# Author: wxm71(weixing.mei@aispeech.com)

import sys

import mxnet as mx
from nce import nce_loss, LMNceIter, NceMetric

def train_sym_gen(args, vocab_size, pad_label):
    cell = mx.rnn.SequentialRNNCell()
    for i in range(args.num_layers):
        cell.add(mx.rnn.LSTMCell(num_hidden=args.num_hidden, prefix='lstm_l%d_'%i))

    def sym_gen(seq_len):
        # [batch_size, seq_len]
        data = mx.sym.Variable('data')

        # map input to a embeding vector
        embedIn = mx.sym.Embedding(data=data, input_dim=vocab_size, output_dim=args.num_embed,name='input_embed')

        # pass embedding vector to lstm
        # [batch_size, seq_len, num_hidden]
        output, _ = cell.unroll(seq_len, inputs=embedIn, layout='NTC', merge_outputs=True)

        # map label to embeding
        label = mx.sym.Variable('label')
        labwgt = mx.sym.Variable('label_weight')

        # define output embeding matrix
        #
        # TODO: change to adapter binding
        embedwgt = mx.sym.Variable(name='output_embed_weight', shape=(vocab_size, args.num_hidden))
        pred = nce_loss(output, label, labwgt, embedwgt, vocab_size, args.num_hidden, args.num_label, seq_len, pad_label)
 
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

        # [seq_len*batch_size, num_hidden]
        outputs, states = stack.unroll(seq_len, inputs=embed, layout='TNC', merge_outputs=True)

        # [seq_len*batch_size, 1, num_hidden] 
        pred = mx.sym.Reshape(data=outputs, shape=(-1, 1, args.num_hidden*(1+args.bidirectional)))

        # get output embedding
        # TODO: this is a constant, initialize it only one-time 
        #
        # [vocab_size] -> [vocab_size, num_hidden]
        allLab = mx.sym.Variable('alllab', shape=(vocab_size,), dtype='float32')
        labs = mx.sym.Embedding(data=allLab, input_dim=vocab_size, output_dim=args.num_hidden, name='output_embed')

        # pred: [seq_len*batch_size, 1, num_hidden] 
        # labs: [vocab_size, num_hidden]
        # output: [seq_len*batch_size, vocab_size, num_hidden]
        pred = mx.sym.broadcast_mul(pred, labs)

        # [seq_len*batch_size, vocab_size]
        pred = mx.sym.sum(data=pred, axis=2)

        label = mx.sym.Variable('label')
        label = mx.sym.Reshape(label, shape=(-1,))

        pred = mx.sym.SoftmaxOutput(data=pred, label=label, name='softmax')

        return pred, ('data',), ('label',)

    return sym_gen

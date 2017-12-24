#!/usr/bin/env python
# coding: utf-8
#
# Usage: 

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

        # project to vocab 
        pred = mx.sym.Reshape(data=output, shape=(-1, args.num_hidden))
        pred = mx.sym.FullyConnected(data=pred, num_hidden=vocab_size, name="project")

        label = mx.sym.Reshape(data=label, shape=(-1,))
        pred = mx.sym.SoftmaxOutput(data=pred, label=label)

        return pred, ('data',), ('label',)
    
        # define output embeding matrix
        #labwgt = mx.sym.Variable('label_weight')
        #embedwgt = mx.sym.Variable(name='output_embed_weight', shape=(vocab_size, args.num_hidden))
        #pred = nce_loss(output, label, labwgt, embedwgt, vocab_size, args.num_hidden, args.num_label, seq_len, pad_label)
        #return pred, ('data',), ('label', 'label_weight')

    return cell, sym_gen



def test_sym_gen(args, vocab_size): 
    if not args.stack_rnn:
        stack = mx.rnn.FusedRNNCell(args.num_hidden, num_layers=args.num_layers,
                mode='lstm', bidirectional=args.bidirectional).unfuse()
    else:
        stack = mx.rnn.SequentialRNNCell()
        for i in range(args.num_layers):
            stack.add(mx.rnn.LSTMCell(num_hidden=args.num_hidden, prefix='lstm_l%d_'%i))

    def sym_gen(seq_len):
        data = mx.sym.Variable('data')
        embed = mx.sym.Embedding(data=data, input_dim=vocab_size, output_dim=args.num_embed, name='input_embed')

        # [batch_size, seq_len, num_hidden]
        pred, states = stack.unroll(seq_len, inputs=embed, layout='NTC', merge_outputs=True)

        # [seq_len*batch_size, 1, num_hidden] 
        #pred = mx.sym.Reshape(data=outputs, shape=(-1, 1, args.num_hidden*(1+args.bidirectional)))

        # get output embedding
        # TODO: this is a constant, initialize it only one-time 
        #
        # [vocab_size] -> [vocab_size, num_hidden]
        # allLab = mx.sym.Variable('alllab', shape=(vocab_size-1,), dtype='float32')
        # labs = mx.sym.Embedding(data=allLab, input_dim=vocab_size, output_dim=args.num_hidden, name='output_embed')

        # # [batch_size, seq_len, 1, num_hidden] 
        # pred = mx.sym.Reshape(data=pred, shape=(-1, 1, args.num_hidden))

        # # labs: [vocab_size, num_hidden]
        # # output: [batch_size*seq_len, vocab_size, num_hidden]
        # pred = mx.sym.broadcast_mul(pred, labs)

        # # [batch_size*seq_len, vocab_size]
        # pred = mx.sym.sum(data=pred, axis=2)
        # pred = mx.sym.sigmoid(data=pred)

        # label = mx.sym.Variable('label')
        # label = mx.sym.Reshape(data=label, shape=(-1,))

        # pred = mx.sym.SoftmaxOutput(data=pred, label=label, name='softmax')

        pred = mx.sym.Reshape(data=pred, shape=(-1, args.num_hidden))
        pred = mx.sym.FullyConnected(data=pred, num_hidden=vocab_size, name="project")

        label = mx.sym.Variable('label') 
        label = mx.sym.Reshape(data=label, shape=(-1,))

        pred = mx.sym.SoftmaxOutput(data=pred,label=label)

        return pred, ('data',), ('label',)

    return sym_gen, stack

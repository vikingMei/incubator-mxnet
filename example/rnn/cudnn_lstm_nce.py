#!/usr/bin/env python
# coding: utf-8

import sys
import argparse

import numpy as np
import mxnet as mx

from mxnet.base import MXNetError

from nce import nce_loss, LMNceIter, NceMetric
from utils import tokenize_text

start_label = 1
invalid_label = 0

buckets = [10, 20, 30, 40, 50, 60, 70, 80]


def train(args):
    layout = 'NT'
    train_sent, vocab, freq = tokenize_text(args.train_data, start_label=start_label, invalid_label=invalid_label)
    val_sent, _, _ = tokenize_text(args.valid_data, vocab=vocab, start_label=start_label, invalid_label=invalid_label)

    # layout, format of data and label. 'NT' means (batch_size, length) and 'TN' means (length, batch_size).
    data_train  = LMNceIter(train_sent, args.batch_size, freq, 
                            layout=layout,
                            buckets=buckets, 
                            invalid_label=invalid_label, 
                            num_label=args.num_label)

    data_val = LMNceIter(val_sent, args.batch_size, freq, 
                            layout=layout,
                            buckets=buckets, 
                            invalid_label=invalid_label, 
                            num_label=args.num_label)

    cell = mx.rnn.SequentialRNNCell()
    for i in range(args.num_layers):
        cell.add(mx.rnn.LSTMCell(num_hidden=args.num_hidden, prefix='lstm_l%d_'%i))

    def sym_gen(seq_len):
        # [batch_size, seq_len]
        data = mx.sym.Variable('data')

        # map input to a embeding vector
        embedIn = mx.sym.Embedding(data=data, input_dim=len(vocab), output_dim=args.num_embed,name='input_embed')

        # pass embedding vector to lstm
        # [batch_size, seq_len, num_hidden]
        output, _ = cell.unroll(seq_len, inputs=embedIn, layout='NTC', merge_outputs=True)
        #output = output.reshape(-1, num_embed)

        # map label to embeding
        label = mx.sym.Variable('label')
        labwgt = mx.sym.Variable('label_weight')

        # define output embeding matrix
        #
        # TODO: change to adapter binding
        embedwgt = mx.sym.Variable(name='output_embed_weight', shape=(len(vocab), args.num_hidden))
        pred = nce_loss(output, label, labwgt, embedwgt, len(vocab), args.num_hidden, args.num_label, seq_len)
 
        return pred, ('data',), ('label', 'label_weight')

    if args.gpus:
        contexts = [mx.gpu(int(i)) for i in args.gpus.split(',')]
    else:
        contexts = mx.cpu(0)

    model = mx.mod.BucketingModule(
        sym_gen             = sym_gen,
        default_bucket_key  = data_train.default_bucket_key,
        context             = contexts)

    if args.load_epoch:
        _, arg_params, aux_params = mx.rnn.load_rnn_checkpoint(
            cell, args.model_prefix, args.load_epoch)
    else:
        arg_params = None
        aux_params = None

    opt_params = {
      'learning_rate': args.lr,
      'wd': args.wd
    }

    if args.optimizer not in ['adadelta', 'adagrad', 'adam', 'rmsprop']:
        opt_params['momentum'] = args.mom

    model.fit(
        train_data          = data_train,
        eval_data           = data_val,
        eval_metric         = NceMetric(invalid_label),
        kvstore             = args.kv_store,
        optimizer           = args.optimizer,
        optimizer_params    = opt_params, 
        initializer         = mx.init.Xavier(factor_type="in", magnitude=2.34),
        arg_params          = arg_params,
        aux_params          = aux_params,
        begin_epoch         = args.load_epoch,
        num_epoch           = args.num_epochs,
        batch_end_callback  = mx.callback.Speedometer(args.batch_size, args.disp_batches),
        epoch_end_callback  = mx.rnn.do_rnn_checkpoint(cell, args.model_prefix, 1)
                              if args.model_prefix else None)

def test(args):
    assert args.model_prefix, "Must specifiy path to load from"

    # generate data iterator
    layout = 'TN'
    train_sent, vocab, _ = tokenize_text(args.train_data, start_label=start_label, invalid_label=invalid_label)
    test_sent, _, _ = tokenize_text(args.test_data, vocab=vocab, start_label=start_label, invalid_label=invalid_label)
    data_test    = mx.rnn.BucketSentenceIter(test_sent, args.batch_size, 
            buckets=buckets,
            invalid_label=invalid_label, 
            label_name='label',
            data_name='data',
            layout=layout)

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
        embed = mx.sym.Embedding(data=data, input_dim=len(vocab), output_dim=args.num_embed, name='input_embed')

        stack.reset()

        # [seq_len*batch_size, num_hidden]
        outputs, states = stack.unroll(seq_len, inputs=embed, layout='TNC', merge_outputs=True)

        # [seq_len*batch_size, 1, num_hidden] 
        pred = mx.sym.Reshape(data=outputs, shape=(-1, 1, args.num_hidden*(1+args.bidirectional)))

        # get output embedding
        # TODO: this is a constant, initialize it only one-time 
        #
        # [vocab_size] -> [vocab_size, num_hidden]
        allLab = mx.sym.Variable('alllab', shape=(len(vocab),), dtype='float32')
        labs = mx.sym.Embedding(data=allLab, input_dim=len(vocab), output_dim=args.num_hidden, name='output_embed')

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

    if args.gpus:
        contexts = [mx.gpu(int(i)) for i in args.gpus.split(',')]
    else:
        contexts = mx.cpu(0)

    # 定义一个模型，使用bucket方式进行训练
    model = mx.mod.BucketingModule(
        sym_gen             = sym_gen,
        default_bucket_key  = data_test.default_bucket_key,
        context             = contexts)

    datashape = data_test.provide_data
    labelshape = data_test.provide_label
    
    model.bind(datashape, labelshape, for_training=False)

    # note here we load using SequentialRNNCell instead of FusedRNNCell.
    _, arg_params, aux_params = mx.rnn.load_rnn_checkpoint(stack, args.model_prefix, args.load_epoch)
    arg_params['alllab'] =  mx.ndarray.arange(len(vocab), dtype='float32').as_in_context(contexts)
    model.set_params(arg_params, aux_params)

    model.score(data_test, mx.metric.Perplexity(invalid_label),
                batch_end_callback=mx.callback.Speedometer(args.batch_size, 5))


if __name__ == '__main__':
    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    parser = argparse.ArgumentParser(description="Train RNN on Penn Tree Bank",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--test', default=False, action='store_true',
            help='whether to do testing instead of training')

    parser.add_argument('--model-prefix', type=str, default=None,
            help='path to save/load model')
    parser.add_argument('--load-epoch', type=int, default=0,
            help='load from epoch')
    parser.add_argument('--num-layers', type=int, default=2,
            help='number of stacked RNN layers')
    parser.add_argument('--num-hidden', type=int, default=200,
            help='hidden layer size')
    parser.add_argument('--num-embed', type=int, default=200,
            help='embedding layer size')
    parser.add_argument('--bidirectional', type=bool, default=False,
            help='whether to use bidirectional layers')
    parser.add_argument('--gpus', type=str,
            help='list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu. ' \
                    'Increase batch size when using multiple gpus for best performance.')

    parser.add_argument('--kv-store', type=str, default='device', help='key-value store type')

    parser.add_argument('--num-epochs', type=int, default=25, help='max num of epochs')

    parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate')

    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer type')

    parser.add_argument('--mom', type=float, default=0.0, help='momentum for sgd')

    parser.add_argument('--wd', type=float, default=0.00001, help='weight decay for sgd')

    parser.add_argument('--batch-size', type=int, default=20, help='the batch size.')

    parser.add_argument('--disp-batches', type=int, default=50, help='show progress for every n batches')

    parser.add_argument('--stack-rnn', default=False, help='stack fused RNN cells to reduce communication overhead')

    parser.add_argument('--dropout', type=float, default='0.0', help='dropout probability (1.0 - keep probability)')

    parser.add_argument('--num-label', type=int, default=20, help='number of label for each input')
    parser.add_argument('--bind-embeding', type=bool, default=False, help='whether bind input and out embeding matrix')

    parser.add_argument('--train-data', type=str, default='./data/ptb.train.txt', help='train data')
    parser.add_argument('--valid-data', type=str, default='./data/ptb.valid.txt', help='valid data')
    parser.add_argument('--test-data', type=str, default='./data/ptb.test.txt', help='test data')

    args = parser.parse_args()

    if args.num_layers >= 4 and len(args.gpus.split(',')) >= 4 and not args.stack_rnn:
        print('WARNING: stack-rnn is recommended to train complex model on multiple GPUs')

    if args.test:
        # Demonstrates how to load a model trained with CuDNN RNN and predict
        # with non-fused MXNet symbol
        test(args)
    else:
        train(args)

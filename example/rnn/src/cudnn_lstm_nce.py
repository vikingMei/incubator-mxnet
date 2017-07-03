#!/usr/bin/env python
# coding: utf-8

import os
import sys
import json
import codecs
import argparse

import numpy as np
import mxnet as mx

from mxnet.base import MXNetError

from lstm_nce import NceMetric, gen_default_buckets
from lstm_nce_sym import train_sym_gen, test_sym_gen
from loader import tokenize_text, get_nce_iter, get_repeat_iter
from utils import get_lradpter

pad_label = 0
invalid_label = 1
start_label = 2

buckets = [10, 20, 30, 40, 50, 60, 70, 80]


def train(args):
    if args.repeat:
        reader = get_repeat_iter 
    else:
        reader = get_nce_iter

    if args.vocab:
        fid = open(args.vocab, 'r')
        vocab = json.load(fid)
        logging.debug('load vocab from [%s]' % args.vocab)
    else:
        vocab = None

    data_train, vocab, freq = reader(args.train_data, start_label, invalid_label, pad_label, 
            args.batch_size, buckets, args.num_label, vocab)

    data_val, _, _ = reader(args.valid_data, start_label, invalid_label, pad_label, 
            args.batch_size, buckets, args.num_label,
            vocab, freq)
    data_val.for_train = False

    # save vocab
    if args.model_prefix:
        pname = os.path.dirname( args.model_prefix)
        pname = '%s/vocab.json' % pname 
        with codecs.open(pname, 'w', 'utf-8') as fid:
            json.dump(vocab, fid)

    if args.gpus:
        contexts = [mx.gpu(int(i)) for i in args.gpus.split(',')]
    else:
        contexts = mx.cpu(0)

    cell,sym_gen = train_sym_gen(args, len(vocab), pad_label, data_train.negdis)
    model = mx.mod.BucketingModule(
        sym_gen             = sym_gen,
        default_bucket_key  = data_train.default_bucket_key,
        context             = contexts)

    if args.load_epoch:
        _, arg_params, aux_params = mx.rnn.load_rnn_checkpoint(
            cell, args.model_prefix, args.load_epoch)
    else:
        arg_params = None
        aux_params = {}

    aux_params['negdis'] = mx.nd.array(data_train.negdis)

    opt_params = {
      'learning_rate': args.lr,
      'wd': args.wd
    }

    if args.optimizer not in ['adadelta', 'adagrad', 'adam', 'rmsprop']:
        opt_params['momentum'] = args.mom

    model.fit(
        train_data          = data_train,
        eval_data           = data_val,
        eval_metric         = NceMetric(args.num_label, data_train.negdis, pad_label, step=args.disp_batches),
        kvstore             = args.kv_store,
        optimizer           = args.optimizer,
        optimizer_params    = opt_params, 
        initializer         = mx.init.Xavier(factor_type="in", magnitude=0.12),
        arg_params          = arg_params,
        aux_params          = aux_params,
        begin_epoch         = args.load_epoch,
        num_epoch           = args.num_epochs,
        #monitor             = mx.mon.Monitor(args.disp_batches, lambda x: x),
        batch_end_callback  = mx.callback.Speedometer(args.batch_size, args.disp_batches, auto_reset=False),
        epoch_end_callback  = mx.rnn.do_rnn_checkpoint(cell, args.model_prefix, 1)
                              if args.model_prefix else None,
        valid_callback      = get_lradpter(model, cell, args.min_epoch, args.model_prefix))


def test(args):
    assert args.model_prefix, "Must specifiy path to load from"

    pname = '%s/vocab.json' % os.path.dirname(args.model_prefix)
    with open(pname, 'r') as fid:
        vocab = json.load(fid)

    test_sent, _, _ = tokenize_text(args.test_data, vocab, start_label=start_label, invalid_label=invalid_label)

    # invlid_label below are used for padding, different from previous(which is <eos>)
    layout = 'NT'
    data_test = mx.rnn.BucketSentenceIter(test_sent, args.batch_size, 
            buckets=buckets,
            invalid_label=pad_label, 
            label_name='label',
            data_name='data',
            layout=layout)

    if args.gpus:
        contexts = [mx.gpu(int(i)) for i in args.gpus.split(',')]
    else:
        contexts = mx.cpu(0)

    sym_gen,cell = test_sym_gen(args, len(vocab)) 

    model = mx.mod.BucketingModule(
        sym_gen             = sym_gen,
        default_bucket_key  = data_test.default_bucket_key,
        context             = contexts)

    model.bind(data_test.provide_data, data_test.provide_label, for_training=False)

    _, arg_params, aux_params = mx.rnn.load_rnn_checkpoint(cell, args.model_prefix, args.load_epoch)
    if args.gpus:
        arg_params['alllab'] =  mx.ndarray.arange(0, len(vocab), dtype='float32').as_in_context(contexts[0])
    else:
        arg_params['alllab'] =  mx.ndarray.arange(0, len(vocab), dtype='float32').as_in_context(contexts)
    model.set_params(arg_params, aux_params)

    ppl = model.score(data_test, mx.metric.Perplexity(pad_label),
                batch_end_callback=mx.callback.Speedometer(args.batch_size, 5, auto_reset=False))

    print('final ppl: ', ppl)


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

    parser.add_argument('--num-epochs', type=int, default=30, help='max num of epochs')

    parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate')

    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer type')

    parser.add_argument('--mom', type=float, default=0.0, help='momentum for sgd')

    parser.add_argument('--wd', type=float, default=0.00001, help='weight decay for sgd')

    parser.add_argument('--batch-size', type=int, default=32, help='the batch size.')

    parser.add_argument('--disp-batches', type=int, default=50, help='show progress for every n batches')

    parser.add_argument('--stack-rnn', default=False, help='stack fused RNN cells to reduce communication overhead')

    parser.add_argument('--dropout', type=float, default='0.0', help='dropout probability (1.0 - keep probability)')
    parser.add_argument('--repeat', action='store_true', default=False, help='using repeat data iter or not')
    parser.add_argument('--lnz', type=float, default=9.0, help='normalization constant')
    parser.add_argument('--min-epoch', type=int, default=4, help='minimize epoch before adjust learning rate')

    parser.add_argument('--num-label', type=int, default=20, help='number of label for each input')
    parser.add_argument('--bind-embeding', type=bool, default=False, help='whether bind input and out embeding matrix')

    parser.add_argument('--train-data', type=str, default='./data/ptb.train.txt', help='train data')
    parser.add_argument('--valid-data', type=str, default='./data/ptb.valid.txt', help='valid data')
    parser.add_argument('--test-data', type=str, default='./data/ptb.test.txt', help='test data')

    parser.add_argument('--vocab', type=str, default=None, help='use pre-generate vocabulary instead of generate from corpus')

    args = parser.parse_args()

    if args.num_layers >= 4 and len(args.gpus.split(',')) >= 4 and not args.stack_rnn:
        print('WARNING: stack-rnn is recommended to train complex model on multiple GPUs')

    if args.test:
        # Demonstrates how to load a model trained with CuDNN RNN and predict
        # with non-fused MXNet symbol
        test(args)
    else:
        train(args)

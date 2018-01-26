# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import pdb
import math
import json
import logging
import argparse

import numpy as np
import mxnet as mx, math
from mxnet.model import BatchEndParam

from model import *
from module import *
from data import Corpus, CorpusIter, NceCorpusIter


parser = argparse.ArgumentParser(description='PennTreeBank LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/ptb.',
                    help='location of the data corpus')
parser.add_argument('--emsize', type=int, default=650,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=650,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=1.0,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.2,
                    help='gradient clipping by global norm')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--log-interval', type=int, default=200,
                    help='report interval')
parser.add_argument('--seed', type=int, default=3,
                    help='random seed')
parser.add_argument('--output', type=str, default='./output')
parser.add_argument('--use-nce', action='store_true', 
                    help='whether use nce loss')
parser.add_argument('--num-label', type=int, default=10,
                    help='numbel of label, include 1 positive label')
parser.add_argument('--lnz', type=int, default=9,
                    help='lnz used to normalization lstm network output')
args = parser.parse_args()

best_loss = 9999

def evaluate(valid_module, data_iter, epoch, mode, bptt, batch_size):
    total_loss = 0.0
    nbatch = 0
    for batch in data_iter:
        valid_module.forward(batch, is_train=False)
        outputs = valid_module.get_loss()
        total_loss += mx.nd.sum(outputs[0]).asscalar()
        nbatch += 1
    data_iter.reset()
    loss = total_loss / bptt / batch_size / nbatch
    logging.info('Iter[%d] %s loss:\t%.7f, Perplexity: %.7f' % (epoch, mode, loss, math.exp(loss)))
    return loss 


def nce_evaluate(valid_module, data_iter, epoch, mode, bptt, batch_size):
    total_loss = 0.0
    nbatch = 0

    total_acc = 0.0
    for batch in data_iter:
        valid_module.forward(batch, is_train=False)

        # [bptt*batch_size, vocab_size]
        pred = valid_module.get_outputs(merge_multi_context=False)[-2]
        pred = pred[0]

        # [bptt,batch_size,num_label]
        # -> [bptt*batch_size]
        label = batch.label[0].reshape((bptt*batch_size,-1))
        label = label[:,0].as_in_context(pred.context)

        pred = mx.nd.pick(pred, label, axis=-1)
        pred = -mx.nd.log(pred+1e-9)
        total_loss += mx.nd.sum(pred).asscalar()
        total_acc += mx.nd.sum(valid_module.get_loss(True)[0]).asscalar()

        nbatch += 1
    data_iter.reset()
    norm = bptt*batch_size*nbatch
    loss = total_loss/norm
    acc = total_acc/norm
    logging.info('Iter[%d] %s loss:\t%.7f, Perplexity: %.7f, Acc: %.7f' % (epoch, mode, loss, math.exp(loss), acc))
    return loss 


if __name__ == '__main__':
    # args
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    args = parser.parse_args()
    logging.info(args)
    ctx = mx.gpu(0)
    batch_size = args.batch_size
    bptt = args.bptt
    mx.random.seed(args.seed)

    # data
    corpus = Corpus(args.data)
    ntokens = len(corpus.dictionary)
    if args.use_nce:
        train_data = NceCorpusIter(corpus.train, batch_size, bptt, args.num_label, corpus.negative)
        valid_data = NceCorpusIter(corpus.valid, batch_size, bptt, args.num_label, corpus.negative)
        test_data = NceCorpusIter(corpus.test, batch_size, bptt, args.num_label, corpus.negative)
        evaluatefunc = nce_evaluate
    else:
        train_data = CorpusIter(corpus.train, batch_size, bptt)
        valid_data = CorpusIter(corpus.valid, batch_size, bptt)
        test_data = CorpusIter(corpus.test, batch_size, bptt)
        evaluatefunc = evaluate


    # save vocab
    corpus.dictionary.dump('%s/vocab.json'%args.output)

    # model
    loss, states, state_names = rnn(bptt, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, batch_size, 
            args.tied, args.use_nce, corpus.negdis, args.lnz)

    label_names = [x[0] for x in train_data.provide_label] 
    data_names = [x[0] for x in train_data.provide_data] 

    # module
    module = CustomStatefulModule(loss, states, data_names=data_names, label_names=label_names,
            state_names=state_names, context=ctx)
    module.bind(data_shapes=train_data.provide_data, label_shapes=train_data.provide_label)
    module.init_params(initializer=mx.init.Xavier())
    optimizer = mx.optimizer.create('sgd', learning_rate=args.lr, rescale_grad=1.0/batch_size)
    module.init_optimizer(optimizer=optimizer)

    # metric
    speedometer = mx.callback.Speedometer(batch_size, args.log_interval)

    # train
    logging.info("Training started ... ")
    for epoch in range(args.epochs):
        # train
        total_loss = 0.0
        nbatch = 0
        for batch in train_data:
            module.forward(batch, is_train=True)
            module.backward()
            module.update(max_norm=args.clip * bptt * batch_size)

            # update metric
            outputs = module.get_loss(args.use_nce)
            total_loss += mx.nd.sum(outputs[0]).asscalar()
            speedometer_param = BatchEndParam(epoch=epoch, nbatch=nbatch,
                                              eval_metric=None, locals=locals())
            speedometer(speedometer_param)
            if nbatch % args.log_interval == 0 and nbatch > 0:
                cur_loss = total_loss / bptt / batch_size / args.log_interval
                logging.info('Iter[%d] Batch [%d]\tLoss:  %.7f,\tPerplexity:\t%.7f' % \
                             (epoch, nbatch, cur_loss, cur_loss))
                             #(epoch, nbatch, cur_loss, math.exp(cur_loss)))
                total_loss = 0.0
            nbatch += 1


        # validation
        valid_loss = evaluatefunc(module, valid_data, epoch, 'Valid', bptt, batch_size)
        if valid_loss < best_loss:
            best_loss = valid_loss
            # test
            test_loss = evaluatefunc(module, test_data, epoch, 'Test', bptt, batch_size)
        else:
            optimizer.lr *= 0.5
        train_data.reset()
        module.save_checkpoint(args.output + '/model', epoch)
    logging.info("Training completed. ")
    module.save_params(args.output +'/final.mdl')

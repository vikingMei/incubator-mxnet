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

import logging
import mxnet as mx

from model import get_ce_net
from loader import Corpus,CorpusIter
from argument import build_arg_parser

def create_eval_end_cb(prefix, reset_param):
    '''
    callback after evaluate on validation data, which call after each epoch

    STEPS:
        1. update optimizer 
        2. update model parameter
        3. create best model link
    '''
    best_metric = 999999 
    best_epoch = 0 

    def _callback(params):
        nonlocal best_metric 
        nonlocal best_epoch

        model = params.locals['self'] 

        name, metric = params.eval_metric.get()
        logging.info('learning rate: %f, [%s] on valid set is [%f]'%(model._optimizer.lr, name, metric))

        if metric>best_metric:
            model._optimizer.lr /= 2.0
            logging.info('update learning rate to %f'%model._optimizer.lr)

            if reset_param:
                fname = '%s-%04d.params' % (prefix,best_epoch)
                logging.info('reset parameter from ' % best_epoch)
                model.load_params(fname)
        else:
            best_metric = metric
            best_epoch = params.epoch

    return _callback


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(message)s')

    parse = build_arg_parser()
    args = parse.parse_args()
    logging.info(args)

    # load corpus
    if args.use_nce:
        corpus = NceCorpus(basedir=args.data)
    else:
        corpus = Corpus(basedir=args.data)
    corpus.vocab.dump('%s/vocab.json'%args.output)

    data_train = corpus.get_train_iter(args.batch_size, args.bptt)
    data_valid = corpus.get_valid_iter(args.batch_size, args.bptt, 1)

    # build network
    network = get_ce_net(args.num_layer, args.num_hidden, args.num_embed, args.bptt, 
            len(corpus.vocab), dropout=args.dropout, tied=args.tied)
    model = mx.mod.Module(symbol=network,
        data_names=[x[0] for x in data_train.provide_data],
        label_names=[y[0] for y in data_train.provide_label],
        context=[mx.gpu(0) if args.gpu else mx.cpu()])

    # get metric
    metric = mx.metric.Perplexity(ignore_label=corpus.vocab.PAD_ID) 

    modprefix = '%s/model'%args.output

    eval_end_cb = create_eval_end_cb(modprefix, False)

    model.fit(
        train_data=data_train,
        num_epoch=args.num_epoch,
        optimizer='sgd',
        optimizer_params={'learning_rate': args.lr, 'rescale_grad': 1.0/args.batch_size},
        initializer=mx.init.Xavier(),
        eval_data=data_valid,
        eval_metric=metric,
        eval_end_callback=eval_end_cb,
        batch_end_callback=mx.callback.Speedometer(args.batch_size, 50),
        epoch_end_callback=mx.callback.do_checkpoint(modprefix))

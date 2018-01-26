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

import nce
import numpy as np
import mxnet as mx

def rnn(bptt, vocab_size, num_embed, nhid,
        num_layers, dropout, batch_size, tied, use_nce, negdis, lnz):
    # encoder
    data = mx.sym.Variable('data')
    weight = mx.sym.var("encoder_weight", init=mx.init.Uniform(0.1))
    embed = mx.sym.Embedding(data=data, weight=weight, input_dim=vocab_size,
                             output_dim=num_embed, name='embed')

    # stacked rnn layers
    states = []
    state_names = []
    outputs = mx.sym.Dropout(embed, p=dropout)
    for i in range(num_layers):
        prefix = 'lstm_l%d_' % i

        cell = mx.rnn.FusedRNNCell(num_hidden=nhid, prefix=prefix, get_next_state=True,
                                   forget_bias=0.0, dropout=dropout)

        outputs, next_states = cell.unroll(bptt, inputs=outputs,
                                           merge_outputs=True, layout='TNC')

        outputs = mx.sym.Dropout(outputs, p=dropout)
        states += next_states

    # decoder  
    # [bptt*batch_size, nhid]
    pred = mx.sym.Reshape(outputs, shape=(-1, nhid))
    if tied:
        assert(nhid == num_embed), "the number of hidden units and the embedding size must batch for weight tying"
    else:
        weight = mx.sym.var("decoder_weight", init=mx.init.Uniform(0.1))

    if use_nce:
        loss = nce_loss(pred, vocab_size, nhid, weight, negdis, lnz)
    else:
        loss = softmax_ce_loss(pred, weight, vocab_size)

    return loss, [mx.sym.stop_gradient(s) for s in states], state_names


def softmax_ce_loss(pred, weight, vocab_size):
    # project hidden to vocab_size distribution  
    # [bptt*batch_size, num_hidden]
    # -> [bptt*batch_size, vocab_size]
    pred = mx.sym.FullyConnected(data=pred, weight=weight, num_hidden=vocab_size, name='pred')
    logits = mx.sym.log_softmax(pred, axis=-1)

    # softmax cross-entropy loss
    # [bptt,batch_size]
    # -> [bptt*batch_size]
    label = mx.sym.Variable('label')
    label = mx.sym.Reshape(label, shape=(-1,))

    # [bptt*batch_size,]
    loss = -mx.sym.pick(logits, label, axis=-1, keepdims=True)
    return mx.sym.make_loss(loss, name='softmax_ce_loss') 


def sigmoid_nce_loss(pred, vocab_size, num_hidden, decode_weight):
    # [bptt,batch_size,num_label]
    # -> [bptt*batch_size,num_label]
    label = mx.sym.Variable('label')
    label = mx.sym.Reshape(label, shape=(-3, 0))

    label_weight = mx.sym.Variable('label_weight')
    label_weight = mx.sym.Reshape(label_weight, shape=(-3, 0))

    # [bptt*batch_size,num_label,num_hidden]
    label_embed = mx.sym.Embedding(label, weight=decode_weight, input_dim=vocab_size,
                             output_dim=num_hidden, name='decode_embed')

    return mx.sym.Custom(op_type='nce_output', pred=pred, label_embed=label_embed, 
            label_weight=label_weight, decode_weight=decode_weight, vocab_size=vocab_size)


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


def nce_loss(hidden, vocab_size, num_hidden, decode_weight, negdis, lnz):
    # [bptt,batch_size,num_label]
    # -> [bptt*batch_size,num_label]
    label = mx.sym.Variable('label')
    label = mx.sym.Reshape(label, shape=(-3, 0))

    label_weight = mx.sym.Variable('label_weight')
    label_weight = mx.sym.Reshape(label_weight, shape=(-3, 0))

    #lnz = mx.sym.Variable('lnz', shape=(1,))

    # [bptt*batch_size,num_label,num_hidden]
    label_embed = mx.sym.Embedding(label, weight=decode_weight, input_dim=vocab_size, output_dim=num_hidden, name='decode_embed')

    negdis = mx.sym.Variable('negdis', shape=(vocab_size,1), dtype='float32', init=MyConstant(negdis.reshape((-1,1))) )
    negdis = mx.sym.BlockGrad(negdis)

    return mx.sym.Custom(op_type='nce_output', pred=hidden, 
            label=label, label_embed=label_embed, label_weight=label_weight, 
            decode_weight=decode_weight, 
            negdis=negdis, vocab_size=vocab_size, lnz=lnz)

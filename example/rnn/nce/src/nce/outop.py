#!/usr/bin/env python
# coding: utf-8
#
# Usage: 
# Author: wxm71(weixing.mei@aispeech.com)

import pdb
import mxnet as mx

@mx.operator.register("nce_output")
class NceOutputProp(mx.operator.CustomOpProp):
    def __init__(self, vocab_size, lnz):
        super(NceOutputProp, self).__init__(need_top_grad=False)
        self.vocab_size = int(vocab_size)
        self.lnz = lnz

    def create_operator(self, ctx, shapes, dtypes):
        return NceOutput(self.lnz, self.vocab_size)

    def list_arguments(self):
        return ['pred', 'label', 'label_embed', 'label_weight', 'decode_weight', 'negdis']

    def list_outputs(self):
        return ['valid_out', 'output']

    def infer_shape(self, in_shape):
        tmp = in_shape[1]
        oshape = [tmp[0], tmp[1]]
        vshape = [tmp[0],self.vocab_size]
        return in_shape, [vshape,oshape], []

    def infer_type(self, in_type):
        otype = in_type[0]
        return in_type, [otype, otype], []

    #def declare_backward_dependency(self, out_grad, in_data, out_data):
    #    deps = []
    #    deps.append(out_data[1])
    #    deps.append(in_data[0])
    #    deps.append(in_data[1])
    #    deps.append(in_data[2])
    #    return deps


class NceOutput(mx.operator.CustomOp):
    def __init__(self, lnz, vocab_size) -> None:
        super(NceOutput, self).__init__()
        self.lnz = float(lnz)
        self.vocab_size = vocab_size

    def forward(self, is_train, req, in_data, out_data, aux):
        # [bptt*batch_size,num_hidden]
        pred = in_data[0]

        # [bptt*batch_size,num_label,num_hidden]
        label_embed = in_data[2]

        # [bptt*batch_size, num_hidde]
        # -> [bptt*batch_size, 1, num_hidde]
        # -> [bptt*batch_size, num_label, num_hidde]
        # -> [bptt*batch_size, num_label]
        pred = pred.expand_dims(axis=-2)
        pred = mx.nd.broadcast_mul(pred, label_embed)
        pred = pred.sum(axis=-1)
        pred -= self.lnz
        pred = pred.exp()
        
        self.assign(out_data[1], req[1], pred)

        # compute softmax output (only for evaluation)
        if not is_train:
            pred = in_data[0]

            # [vocab_size, num_hidden]
            decode_weight = in_data[4]

            # [bptt*batch_size, vocab_size]
            out = mx.nd.dot(pred, decode_weight.T)
            out = out.softmax(axis=-1)
            self.assign(out_data[0], req[0], out)


    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        # [bptt*batch_size, num_hidden]
        pred = in_data[0] 

        # [bptt*batch_size, num_label]
        pred_out = out_data[1] 

        # [bptt*batch_size, num_label]
        label = in_data[1]

        # [bptt*batch_size, num_label, num_hidden]
        label_embed = in_data[2]

        # [bptt*batch_size, num_label]
        label_weight = in_data[3]

        negdis = in_data[5]

        num_label = label_weight.shape[1]
        num_hidden = label_embed.shape[2]

        pn_label = mx.nd.Embedding(label, negdis, input_dim=self.vocab_size, output_dim=1)
        pn_label = pn_label.reshape((-1, num_label))

        grad_norm = pred_out + (num_label-1)*pn_label
        grad_base = label_weight*(num_label-1)*pn_label - (1-label_weight)*pred_out
        grad_base = -grad_base/grad_norm

        # grad to pred [bptt*batch_size,num_hidden] 
        # [bptt*batch_size, num_label]
        # -> [bptt*batch_size, num_label, 1]
        # -> [bptt*batch_size, num_label, num_hidden]
        # -> [bptt*batch_size, num_hidden]
        grad = grad_base.expand_dims(axis=-1)
        grad = mx.nd.broadcast_mul(grad, label_embed)
        grad = grad.sum(axis=-2)
        self.assign(in_grad[0], req[0], grad)

        # grad to label embed [bptt*batch_size, num_label, num_hidden]
        # [bptt*batch_size, num_label]
        # -> [bptt*batch_size, num_label, num_hidden]
        grad = grad_base.expand_dims(axis=-1)
        grad = grad.broadcast_to(label_embed.shape)

        # [bptt*batch_size, num_hidden]
        # -> [bptt*batch_size, num_label, num_hidden]
        pred = pred.expand_dims(axis=-2)
        pred = pred.broadcast_to(label_embed.shape)

        grad *= pred

        self.assign(in_grad[2], req[2], grad)

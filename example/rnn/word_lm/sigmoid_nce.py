#!/usr/bin/env python
# coding: utf-8
#
# Usage: 
# Author: viking(auimoviki@gmail.com)

import pdb
import mxnet as mx

@mx.operator.register("sigmoid_nce_output")
class SigmoidNceOutputProp(mx.operator.CustomOpProp):
    def __init__(self, vocab_size):
        super(SigmoidNceOutputProp, self).__init__(need_top_grad=False)
        self.vocab_size = int(vocab_size)

    def create_operator(self, ctx, shapes, dtypes):
        return SigmoidNceOutput()

    def list_arguments(self):
        return ['pred', 'label_embed', 'label_weight', 'decode_weight']

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

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        deps = []
        deps.append(out_data[1])
        deps.append(in_data[0])
        deps.append(in_data[1])
        deps.append(in_data[2])
        return deps


class SigmoidNceOutput(mx.operator.CustomOp):
    def __init__(self) -> None:
        super(SigmoidNceOutput, self).__init__()

    def forward(self, is_train, req, in_data, out_data, aux):
        # [bptt*batch_size,num_hidden]
        pred = in_data[0]

        # [bptt*batch_size,num_label,num_hidden]
        label_embed = in_data[1]

        # [vocab_size, num_hidden]
        decode_weight = in_data[3]

        # [bptt*batch_size, num_hidde]
        # -> [bptt*batch_size, 1, num_hidde]
        # -> [bptt*batch_size, num_label, num_hidde]
        # -> [bptt*batch_size, num_label]
        pred = pred.expand_dims(axis=-2)
        pred = mx.nd.broadcast_mul(pred, label_embed)
        pred = pred.sum(axis=-1)
        
        out = 1/(1+mx.nd.exp(-pred))
        self.assign(out_data[1], req[1], out)

        # compute softmax output (only for evaluation)
        if not is_train:
            pred = in_data[0]
            # [bptt*batch_size, vocab_size]
            out = mx.nd.dot(pred, decode_weight.T)
            out = out.softmax(axis=-1)
            self.assign(out_data[0], req[0], out)


    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        # [bptt*batch_size, num_hidden]
        pred = in_data[0] 

        # [bptt*batch_size, num_label]
        out = out_data[1] 

        # [bptt*batch_size, num_label, num_hidden]
        label_embed = in_data[1]

        # [bptt*batch_size, num_label]
        label_weight = in_data[2]

        num_label = label_weight.shape[1]
        num_hidden = label_embed.shape[1]

        # [bptt*batch_size,num_label]
        base = out - label_weight
        base /= num_label 

        # grad to pred [bptt*batch_size,num_hidden] 
        # [bptt*batch_size, num_label]
        # -> [bptt*batch_size, num_label, 1]
        # -> [bptt*batch_size, num_label, num_hidden]
        # -> [bptt*batch_size, num_hidden]
        grad = base.expand_dims(axis=-1)
        grad = mx.nd.broadcast_mul(grad, label_embed)
        grad = grad.sum(axis=-2)
        self.assign(in_grad[0], req[0], grad)

        # grad to label embed [bptt*batch_size, num_label, num_hidden]
        # [bptt*batch_size, num_label]
        # -> [bptt*batch_size, num_label, num_hidden]
        grad = base.expand_dims(axis=-1)
        grad = grad.broadcast_to(label_embed.shape)

        # [bptt*batch_size, num_hidden]
        # -> [bptt*batch_size, num_label, num_hidden]
        pred = pred.expand_dims(axis=-2)
        pred = pred.broadcast_to(label_embed.shape)

        grad *= pred

        self.assign(in_grad[1], req[1], grad)

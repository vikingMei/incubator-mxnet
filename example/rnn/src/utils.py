#!/usr/bin/env python
# coding: utf-8
#
# Usage: 

import os
import mxnet as mx

def get_lradpter(model, cell, min_epoch, prefix):
    model.__lradp_curloss = 0.0
    model.__lradp_best_epoch = 0
    def valcb(epoch, symbol, arg, aux, res):
        '''
        callback on epoch end

        adjust learning rate accoring to current batch learning result
        '''
        oldloss = model.__lradp_curloss
        for name, loss in res:
            lossname = name
            model.__lradp_curloss = loss
            break
        curloss = model.__lradp_curloss

        if epoch<=min_epoch:
            model.__lradp_best_epoch = epoch
            return

        if curloss-oldloss>1e-3:
            # if new loss bigger than older loss, reload previous model
            print("current loss: %f bigger than previous loss: %f, reload previous param" % (curloss, oldloss) )
            fname = '%s-%04d.params' % (prefix, epoch)
            os.rename(fname, '%s-%04d-fail.param'  % (prefix, epoch))

            print('reload previous model: %s-%04d' % (prefix, model.__lradp_best_epoch))
            _, prearg, preaux = mx.rnn.load_rnn_checkpoint(cell, prefix, model.__lradp_best_epoch)
            for k,v in prearg.items():
                arg[k] = v
            for k,v in preaux.items():
                aux[k] = v

            # update learning rate
            model._curr_module._optimizer.lr /= 2.0
            print("update learning rate to %f" % model._curr_module._optimizer.lr)
        else:
            model.__lradp_best_epoch = epoch

    return valcb

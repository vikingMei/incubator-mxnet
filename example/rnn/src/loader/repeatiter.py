#!/usr/bin/env python
# coding: utf-8
#
# Usage: 
# Author: Summer Qing(qingyun.wu@aispeech.com)

import mxnet as mx

class RepeatIter(mx.rnn.BucketSentenceIter):
    def reset(self):
        super(RepeatIter, self).reset()

        repeat = self.idx[0]
        for i in range(0, len(self.idx)):
            self.idx[i] = repeat


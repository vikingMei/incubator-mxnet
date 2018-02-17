#!/usr/bin/env python
# coding: utf-8
#
# Usage: 
# Author: wxm71(weixing.mei@aispeech.com)

import pdb
import logging
import threading
import multiprocessing

import numpy as np
import mxnet as mx

from .utils import batchify


class CorpusIter(mx.io.DataIter):
    def __init__(self, source, batch_size, bptt, num_sub=2):
        super(CorpusIter, self).__init__()
        self.logger = logging.getLogger(str(self.__class__))

        # [nbatch, batch_size]
        self.source_ = batchify(np.array(source), batch_size)

        self.bptt = bptt
        self.batch_size = batch_size 
        self.num_sub = num_sub

        self.ppoll = []
        self.jobend_pid = None
        self.running = False

        self.provide_label = [mx.io.DataDesc(name='label', shape=(bptt, batch_size))]
        self.provide_data = [mx.io.DataDesc(name='data', shape=(bptt, batch_size))]

        self.sub_len = self.get_sub_len() 
        self.queue = multiprocessing.Queue(20)


    def get_num_batch(self):
        return int(self.source_.shape[0]/self.bptt)


    def get_num_sub(self):
        nbatch = self.get_num_batch()
        return nbatch if nbatch<self.num_sub else self.num_sub 


    def get_sub_len(self):
        '''
        get the number of sample(index self.source_ at index 0) processed in each subprocess
        '''
        nbatch = self.get_num_batch()
        nsub = self.get_num_sub()
        return int(nbatch/nsub)*self.bptt


    @staticmethod
    def subproc_tgtfunc(source, idxbeg, idxend, queue, inst):
        '''
        target funtion run in sub-process
        '''
        bptt = inst.bptt

        idx = idxbeg
        idxend = idxend-bptt-1
        while idx<idxend:
            data = source[idx:idx+bptt,:]
            label = source[idx+1:idx+bptt+1, :]
            queue.put((data, label))
            idx += bptt


    @staticmethod
    def jobend_tgtfunc(ppoll, queue):
        for pid in ppoll:
            pid.join()
        queue.put(None)


    def _start(self):
        if self.running:
            self.logger.warning('iter is running, just skip')
            return 
        else:
            flag = False
            for t in self.ppoll:
                flag = t.is_alive()
                if flag: 
                    break
            if flag:
                self.logger.warning('exist alive sub-process, skip corpus iter start')

        self.running = True

        self.ppoll = []

        numsub = self.get_num_sub()
        sublen = self.get_sub_len()

        idxbeg = 0
        idxend = len(self.source_)-1
        for i in range(0, numsub-1): 
            pid = multiprocessing.Process(target=self.__class__.subproc_tgtfunc, 
                    args=(self.source_, idxbeg, idxbeg+sublen+1, self.queue, self))
            pid.start()
            self.ppoll.append(pid)
            idxbeg += sublen

        pid = multiprocessing.Process(target=self.__class__.subproc_tgtfunc, 
                args=(self.source_, idxbeg, idxend, self.queue, self))
        pid.start()
        self.ppoll.append(pid)

        # start job end thread
        self.jobend_pid = threading.Thread(target=self.__class__.jobend_tgtfunc, args=(self.ppoll, self.queue))
        self.jobend_pid.start()


    def _stop(self):
        pass


    def getdata(self):
        return self._next_data


    def getlabel(self):
        return self._next_label


    def iter_next(self):
        batchdata = self.queue.get()
        if batchdata is None:
            return False
        else:
            self._next_data = [ mx.nd.array(batchdata[0]) ]
            self._next_label = [ mx.nd.array(batchdata[1]) ] 
        return True


    def next(self):
        if not self.running:
            self._start()

        if self.iter_next():
            batch = mx.io.DataBatch(data=self.getdata(), provide_data=self.provide_data,
                    label=self.getlabel(), provide_label=self.provide_label)
            return batch
        else:
            self.running = False
            raise StopIteration


    def reset(self):
        self._stop()

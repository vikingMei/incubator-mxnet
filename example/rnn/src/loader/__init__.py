# coding: utf-8
from .loader import tokenize_text 
from .loader import get_nce_iter

import os
import json
import numpy as np

def get_repeat_iter(fname, start_label, invalid_label, pad_label, batch_size, buckets, num_label, vocab=None, freq=None):
    np.random.seed(0)

    dataiter, vocab, freq =  get_nce_iter(fname, start_label, invalid_label, pad_label, batch_size, buckets, num_label, vocab, freq, rand=False)

    for i in range(0, len(dataiter.idx)):
        dataiter.idx[i] = (0, 10)

    return dataiter, vocab, freq

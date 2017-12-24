#!/usr/bin/env python
# coding: utf-8
#
# Usage: find the bucketing by danamic progamming(experiment version)
#

import numpy as np

cntdict = {2: 137, 3: 282, 4: 309, 5: 434, 6: 504, 7: 630, 8: 739, 9: 929, 10: 992, 11: 1101, 12: 1226, 13: 1508, 14: 1457, 15: 1540, 16: 1529, 17: 1640, 18: 1616, 19: 1714, 20: 1580, 21: 1666, 22: 1599, 23: 1566, 24: 1566, 25: 1393, 26: 1452, 27: 1365, 28: 1245, 29: 1146, 30: 1102, 31: 1031, 32: 881, 33: 768, 34: 696, 35: 581, 36: 548, 37: 483, 38: 441, 39: 384, 40: 352, 41: 263, 42: 252, 43: 237, 44: 184, 45: 159, 46: 121, 47: 99, 48: 89, 49: 74, 50: 58, 51: 56, 52: 62, 53: 40, 54: 27, 55: 29, 56: 27, 57: 19, 58: 28, 59: 9, 60: 14, 61: 17, 62: 13, 63: 9, 64: 8, 65: 2, 66: 1, 67: 2, 68: 2, 69: 5, 70: 6, 71: 2, 72: 1, 73: 1, 74: 6, 75: 1, 76: 3, 77: 1, 78: 3, 79: 2, 81: 2, 82: 1, 83: 1}

sorted(cntdict)

batch_size = 40
keys = cntdict.keys()
lenkey = len(keys)

# throwsum[i,j]: cntdict[i+1] + cntdict[i+1] + ... cntdict[j] 
throwsum = np.zeros((lenkey, lenkey)) 
for i in range(0, lenkey-1):
    for j in range(i+1, lenkey):
        key = keys[j]
        throwsum[i,j] = throwsum[i,j-1] + cntdict[key] 

throwsum.tofile('throwsum', sep='\n')

# patsum[i,j]: total pades while padding keys[i], keys[i+1], ..., keys[j-1] to keys[j]
#
# patsum[i,i] = 0
# patsum[i, i+1] = (keys[i+1]-keys[i])*cnt[i+1]
# patsum[i, i+2] = (keys[i+2]-keys[i])*cnt[i+2) + patsum[i, i+1) 
# =>  padnt[i,j] = patsum[i,j-1] + (keys[j]-keys[i])*cnt[j]
patsum = np.zeros((lenkey, lenkey))
for i in range(lenkey-1, -1, -1):
    patsum[i,i] = 0
    keyi = keys[i]
    for j in range(i, lenkey):
        keyj = keys[j]
        patsum[i,j] = patsum[i,j-1] + (keyj-keyi)*cntdict[keyj]

patsum.tofile('patsum', sep='\n')


# throwbuf[i,j]: i is bucket, j is bucket, j<i, and j<k<i is not bucket
# throwbuf[i,i]: i is bucket, j<i is not bucket
# 
# throwbuf[i,i] = throwsum[0,i] + cnt[0]
# throwbuf[i,j] = min(throwbuf[j, :]) + throwsum[j, i]
throwbuf = np.zeros((lenkey, lenkey))

# padbuf[i,j]: i,j same with throwbuf
# 
# padbuf[i,i]:  
padbuf = np.zeros((lenkey, lenkey))

for i in range(0, lenkey-1):
    throwbuf[i,i] = (throwsum[0, i] + cntdict[keys[0]])%batch_size
    padbuf[i,i] = patsum[0, i]

    for j in range(i-1, -1, -1):
        key = keys[j] 
        # find minimum in throwbuf[j,:]
        minidx = np.argmin(throwbuf[j,:j+1])
        minval = throwbuf[j, minidx]
        throwbuf[i, j] = throwsum[j,i]%batch_size+minval

        # find minimum in padbuf[j, :]
        minidx = np.argmin(padbuf[j, :j+1])
        minval = padbuf[j,minidx]
        padbuf[i,j] = patsum[j, i] + minval

throwbuf.tofile('throwbuf', sep='\n')

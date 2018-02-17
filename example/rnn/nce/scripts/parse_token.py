#!/usr/bin/env python
# coding: utf-8
#
# Usage: 
# Author: wxm71(weixing.mei@aispeech.com)

import re
import sys
import pdb
import json

if len(sys.argv)<2:
    print("USAGE: %s vocab fin" % sys.argv[0])
    sys.exit(0)

vocab = sys.argv[1]

fid = open(vocab, 'r')
data = json.load(fid)
fid.close()

idxvocab = {}
for k,v in data.items():
    idxvocab[v] = k

pat = re.compile(r'\.0\b')

fname = sys.argv[2]
for fname in sys.argv[2:]:
    with open(fname) as fid:
        for line in fid:
            line = line.strip().replace('.0', '')
            line = pat.sub('', line)
            line = re.split('[, \t]+', line)
            
            wrds = []
            for x in line:
                if x.startswith('-'):
                    x = float(x)
                    wrds.append('%12.6f'%x)
                else:
                    x = int(x)
                    wrds.append(idxvocab[x])

            print ' '.join(wrds)

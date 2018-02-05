#!/usr/bin/env python
# coding: utf-8
#
# Usage: 
# Author: wxm71(weixing.mei@aispeech.com)

import sys
import argparse


def build_arg_parser():
    '''
    build argument parser
    '''
    parser = argparse.ArgumentParser(description='lstm language model')

    add_data_args(parser)
    add_model_args(parser)
    add_optmizer_args(parser)
    add_nce_args(parser)
    add_log_args(parser)

    parser.add_argument('--seed', type=int, default=3, help='random seed')
    parser.add_argument('--output', type=str, default='./output')

    return parser

def add_model_args(parser):
    args= parser.add_argument_group('MODEL')
    args.add_argument('--num-layer', type=int, default=2, help='number of layers')
    args.add_argument('--num-embed', type=int, default=650, help='size of word embeddings')
    args.add_argument('--tied', action='store_true', help='tie the word embedding and softmax weights')
    args.add_argument('--num-hidden', type=int, default=650, help='number of hidden units per layer')
    args.add_argument('--dropout', type=float, default=0.5, help='dropout applied to layers (0 = no dropout)')


def add_nce_args(parser):
    args = parser.add_argument_group('NCE')
    args.add_argument('--use-nce', action='store_true', help='whether use nce loss')
    args.add_argument('--num-label', type=int, default=10, help='numbel of label used in nce, include 1 positive label')
    args.add_argument('--lnz', type=int, default=9, help='lnz used to normalization lstm network output')


def add_optmizer_args(parser):
    args = parser.add_argument_group('OPTIMIZER')
    args.add_argument('--lr', type=float, default=1.0, help='initial learning rate')
    args.add_argument('--momentum', type=float, default=0.9, help='momentum used in training')
    args.add_argument('--weight-decay', type=float, default=0.0, help='weight decay used in training')
    args.add_argument('--clip', type=float, default=0.2, help='gradient clipping by global norm')
    args.add_argument('--num-epoch', type=int, default=40, help='upper epoch limit')
    args.add_argument("--gpu", action="store_true", dest="gpu", default=False, help="use gpu")


def add_data_args(parser):
    args = parser.add_argument_group('DATA')
    args.add_argument('--data', type=str, default='./data/ptb', help='location of the data corpus')
    args.add_argument('--batch_size', type=int, default=32, help='batch size')
    args.add_argument('--bptt', type=int, default=35, help='max sequence length')


def add_log_args(parser):
    args = parser.add_argument_group('LOG')
    args.add_argument('--log-interval', type=int, default=200, help='report interval')


if '__main__'==__name__:
    parser = build_arg_parser()
    args = parser.parse_args()
    print(args)

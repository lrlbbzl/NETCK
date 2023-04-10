import torch
import os
import argparse
import warnings
import random

parser = argparse.ArgumentParser(description="NETCK")

parser.add_argument('--pretrained-model', default='bert-base-uncased', type=str, help='path to pretrained model')
parser.add_argument('--task', default='WN18RR', type=str, help='dataset name')
parser.add_argument('--train-path', default='./data/WN18RR/train.txt.json', help='train dataset path')
parser.add_argument('--entity-path', default='./data/WN18RR/entities.json', help='entity information path')
parser.add_argument('--max-tokens', default=100, help='max tokens num in encoding')
parser.add_argument('--max-triples', default=5, help='max sampled related triples to central entity')

# hyperparameter in training phase
parser.add_argument('--batch-size', default=512, help='batch size in training')
parser.add_argument('--tau', default=20, help='temperature coefficient in contrastive learning')

args = parser.parse_args()
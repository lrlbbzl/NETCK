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
parser.add_argument('--save-dir', default='./checkpoints', type=str, help='path to save model')
parser.add_argument('--max-tokens', default=100, help='max tokens num in encoding')
parser.add_argument('--max-triples', default=5, help='max sampled related triples to central entity')

# hyperparameter in model
parser.add_argument('--batch-size', default=512, type=int, help='batch size in training')
parser.add_argument('--tau', default=20, help='temperature coefficient in contrastive learning')
parser.add_argument('--hidden-size', default=768, type=int, help='hidden size in pretrained language model')
parser.add_argument('--margin', default=0.02, type=float, help='additive margin for InfoNCE')

# train setting
parser.add_argument('--lr', default=2e-5, type=float, help='learining rate in training')
parser.add_argument('--weight-decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--epoch', default=15, type=int, help='train epoch')
parser.add_argument('--lr-scheduler', default='cosine', type=str, help='type of learning rate scheduler')
parser.add_argument('--warmup-step', default=200, type=int, help='warm up step in optimizer')
parser.add_argument('--loader-workers', default=1, type=1, help='number of workers when loading dataset')
parser.add_argument('--use-amp', default=False, type=bool, help='use automatic mixed precision during training')
parser.add_argument('--grad-clip', default=10.0, type=float, help='clip size for model parameters')
parser.add_argument('--print-freq', default=20, type=int, help='metric print frequency during training')
parser.add_argument('--eval-freq', default=5, type=int, help='evaluate frequency during training')

# test setting
parser.add_argument('--model-path', default='checkpoints/epoch_5.mdl', help='path of model used to evaluate on test dataset')
parser.add_argument('--test-path', default='./data/WN18RR/test.txt.json', help='path of test dataset')

args = parser.parse_args()
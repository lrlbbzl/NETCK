import os
import json
import torch
import random
import torch.utils.data.dataset
from typing import Optional, List
from config import args
from transformers import AutoModel, AutoTokenizer

from structure import EntityDict, EntityItem, TotalGraph
from logger_config import logger

entityDict = EntityDict(args.entity_path)
totalGraph = TotalGraph(args.train_path)
related_triples = totalGraph.get_related_triples()

class Triplet(object):
    def __init__(self, head_id, relation, tail_id):
        self.head_id = head_id
        self.relation = relation
        self.tail_id = tail_id
        self.head_desc = entityDict.id2entity[self.head_id].entity_desc
        self.tail_desc = entityDict.id2entity[self.tail_id].entity_desc
        self.head = entityDict.id2entity[self.head_id].entity
        self.tail = entityDict.id2entity[self.tail_id].entity

        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)

    def split_word(self, words: str):
        if args.task == 'WN18RR':
            return ' '.join(words.split('_')[:-2])
        return words
    
    def tokenize(self, text: str, text_pair: Optional[str] = None):
        encoded_input = self.tokenizer(text=text, text_pair=text_pair if text_pair else None, add_special_tokens=True,
                                       max_length=args.max_tokens, return_token_type_ids=True, truncation=True, return_tensors='pt')
        return encoded_input

    def handle(self, ):
        # relation-aware head 
        head = self.split_word(self.head)
        head_desc = head + ': ' + self.head_desc
        hr_encoded_input = self.tokenize(head_desc, self.relation)

        # tail 
        tail = self.split_word(self.tail)
        tail_desc = tail + ': ' + self.tail_desc
        tail_encoded_input = self.tokenize(tail_desc) 

        # related triples 
        triple_tuples = related_triples[self.head_id]
        random.shuffle(triple_tuples)
        triple_tuples = triple_tuples[:args.max_triples]
        
        triples_desc = [self.split_word(entityDict.id2entity[x[0]].entity) + ' ' + x[1] + ' [MASK].' for x in triple_tuples]
        triples_desc = ' [SEP] '.join(triples_desc)
        triples_encoded_input = self.tokenize(triples_encoded_input)

        return_dict = {
            'hr_input_ids': hr_encoded_input['input_ids'],
            'hr_token_type_ids': hr_encoded_input['token_type_ids'],
            'tail_input_ids': tail_encoded_input['tail_ids'],
            'tail_token_type_ids': tail_encoded_input['token_type_ids'],
            'triples_input_ids': triples_encoded_input['input_ids'],
            'triples_token_type_ids': triples_encoded_input['token_type_ids'],
            'obj': self,
        }
        return return_dict

class Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, path_list):
        self.path_list = path_list
        self.data = []
        for path in path_list:
            self.data.extend(self.load_data(path))

    def load_data(self, path: str, add_reciprocal: bool = True):
        data = json.load(open(path, 'r', encoding='utf-8'))
        logger.info('Load {} triples from {}.'.format(len(data), path))
        samples = []
        for i, sample in enumerate(data):
            samples.append(Triplet(sample['head_id'], sample['relation'], sample['tail_id']))
            if add_reciprocal:
                samples.append(Triplet(sample['tail_id'], 'reverse ' + sample['relation'], sample['tail_id']))
        return samples
    
    def __len__(self, ):
        return len(self.data)
    
    def __getitem__(self, index) :
        return self.data[index].handle()
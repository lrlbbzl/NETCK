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
tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)

class Triplet(object):
    def __init__(self, head_id=None, relation=None, tail_id=None):
        self.head_id = head_id
        self.relation = relation
        self.tail_id = tail_id
        self.head_desc = entityDict.id2entity[self.head_id].entity_desc if head_id else ''
        self.tail_desc = entityDict.id2entity[self.tail_id].entity_desc if tail_id else ''
        self.head = entityDict.id2entity[self.head_id].entity if head_id else ''
        self.tail = entityDict.id2entity[self.tail_id].entity if tail_id else ''

        self.tokenizer = tokenizer

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
        triple_tuples = related_triples.get(self.head_id, [])
        random.shuffle(triple_tuples)
        triple_tuples = triple_tuples[:args.max_triples]
        
        triples_desc = [self.split_word(entityDict.id2entity[x[0]].entity) + ' ' + x[1] + ' [MASK].' for x in triple_tuples]
        triples_desc = ' [SEP] '.join(triples_desc)
        triples_encoded_input = self.tokenize(triples_desc)

        return_dict = {
            'hr_input_ids': hr_encoded_input['input_ids'],
            'hr_token_type_ids': hr_encoded_input['token_type_ids'],
            'tail_input_ids': tail_encoded_input['input_ids'],
            'tail_token_type_ids': tail_encoded_input['token_type_ids'],
            'triples_input_ids': triples_encoded_input['input_ids'],
            'triples_token_type_ids': triples_encoded_input['token_type_ids'],
            'obj': self,
        }
        return return_dict

class Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, path_list, data=None):
        self.path_list = path_list
        if data != None:
            self.data = data
        else:
            self.data = []
            for path in path_list:
                self.data.extend(self.load_data(path, add_forward=True, add_reciprocal=True))

    def load_data(self, path: str, add_forward: bool = True, add_reciprocal: bool = True):
        logger.info('Open: {}.'.format(path))
        data = json.load(open(path, 'r', encoding='utf-8'))
        logger.info('Load {} triples from {}.'.format(len(data), path))
        samples = []
        for i, sample in enumerate(data):
            if add_forward:
                samples.append(Triplet(sample['head_id'], sample['relation'], sample['tail_id']))
            if add_reciprocal:
                samples.append(Triplet(sample['tail_id'], 'reverse ' + sample['relation'], sample['tail_id']))
        return samples
    
    def __len__(self, ):
        return len(self.data)
    
    def __getitem__(self, index) :
        return self.data[index].handle()


def collate(batch_data: List[dict]):
    pad_token = tokenizer.pad_token_id
    hr_input_ids, hr_mask = pad_and_mask([x['hr_input_ids'] for x in batch_data], pad_token=pad_token)
    hr_token_type_ids = pad_and_mask([x['hr_token_type_ids'] for x in batch_data], pad_token=pad_token, need_mask=False)
    tail_input_ids, tail_mask = pad_and_mask([x['tail_input_ids'] for x in batch_data], pad_token=pad_token)
    tail_token_type_ids = pad_and_mask([x['tail_token_type_ids'] for x in batch_data], pad_token=pad_token, need_mask=False)
    triples_input_ids, triples_mask = pad_and_mask([x['triples_input_ids'] for x in batch_data], pad_token=pad_token)
    triples_token_type_ids = pad_and_mask([x['triples_token_type_ids'] for x in batch_data], pad_token=pad_token, need_mask=False)
    
    new_batch_data = {
        'hr_input_ids' : hr_input_ids,
        'hr_mask' : hr_mask,
        'hr_token_type_ids' : hr_token_type_ids,
        'tail_input_ids' : tail_input_ids,
        'tail_mask' : tail_mask,
        'tail_token_type_ids' : tail_token_type_ids,
        'triples_input_ids' : triples_input_ids,
        'triples_mask' : triples_mask,
        'triples_token_type_ids' : triples_token_type_ids,
        'batch_samples' : [x['obj'] for x in batch_data],
    }
    return new_batch_data
    

def pad_and_mask(data, pad_token, need_mask=True):
    mx_length = max([x.shape[1] for x in data])
    batch_size = len(data)
    tokens = torch.LongTensor(batch_size, mx_length).fill_(pad_token)
    if need_mask:
        mask = torch.zeros(*(batch_size, mx_length), dtype=torch.int64)
    for i, item in enumerate(data):
        item = item[0]
        tokens[i, : len(item)].copy_(item)
        if need_mask:
            mask[i, : len(item)].fill_(1)
    if need_mask:
        return tokens, mask
    else:
        return tokens

if __name__ == "__main__":
    print(1)



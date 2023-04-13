import torch
from torch import nn
import torch.nn.functional as F
import json
import pickle
import os
import time

from transformers import AutoModel, AutoTokenizer, AutoConfig
from logger_config import logger

class NETCK(nn.Module):
    def __init__(self, args, ):
        super(NETCK, self).__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained(args.pretrained_model)
        self.batch_size = args.batch_size
        self.tokenzier = AutoTokenizer.from_pretrained(args.pretrained_model)
        self.hr_bert = AutoModel.from_pretrained(args.pretrained_model)
        self.triple_bert = AutoModel.from_pretrained(args.pretrained_model)
        self.tail_bert = AutoModel.from_pretrained(args.pretrained_model)
        self.alpha = nn.Parameter(torch.tensor((1.0 / args.tau)).log(), requires_grad=True)
        self.loss_func = nn.BCEWithLogitsLoss()
        self.W = nn.Parameter(torch.zeros(args.hidden_size, args.hidden_size))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

    def hr_encode(self, input_ids, mask, token_type_ids):
        outputs = self.hr_bert(input_ids=input_ids,
                          attention_mask=mask,
                          token_type_ids=token_type_ids,
                          return_dict=True)
        last_hidden_state = outputs['last_hidden_state']
        bs, length = input_ids.size(0), input_ids.size(1)
        head_tokens = token_type_ids ^ mask
        head_tokens[:, 0] = 0
        token_num = head_tokens.sum(dim=1).unsqueeze(-1)
        head_tokens = head_tokens.unsqueeze(-1).expand(bs, length, last_hidden_state.shape[-1])
        head_embedding = (head_tokens * last_hidden_state).sum(dim=1) / token_num

        rel_tokens = token_type_ids & mask
        token_num = rel_tokens.sum(dim=1).unsqueeze(-1)
        rel_tokens = rel_tokens.unsqueeze(-1).expand(bs, length, last_hidden_state.shape[-1])
        rel_embedding = (rel_tokens * last_hidden_state).sum(dim=1) / token_num
        return head_embedding, rel_embedding # (bs, hidden_size)
    
    def tail_encode(self, input_ids, mask, token_type_ids):
        outputs = self.tail_bert(input_ids=input_ids,
                          attention_mask=mask,
                          token_type_ids=token_type_ids,
                          return_dict=True)
        last_hidden_state = outputs['last_hidden_state']
        bs, length = input_ids.size(0), input_ids.size(1)
        token_num = mask.sum(dim=1).unsqueeze(-1)
        mask = mask.unsqueeze(-1).expand(bs, length, last_hidden_state.shape[-1])
        tail_embedding = (mask * last_hidden_state).sum(dim=1) / token_num
        return tail_embedding # (bs, hidden_size)
    
    def triples_encode(self, input_ids, mask, token_type_ids):
        outputs = self.tail_bert(input_ids=input_ids,
                          attention_mask=mask,
                          token_type_ids=token_type_ids,
                          return_dict=True)
        last_hidden_state = outputs['last_hidden_state']
        bs, length = input_ids.size(0), input_ids.size(1)
        mask = (input_ids == 103).float()
        token_num = mask.sum(dim=1).unsqueeze(-1)
        mask = mask.unsqueeze(-1).expand(bs, length, last_hidden_state.shape[-1])
        triples_embedding = (mask * last_hidden_state).sum(dim=1) / token_num
        return triples_embedding
    
    def forward(self, hr_input_ids, hr_mask, hr_token_type_ids, 
                tail_input_ids, tail_mask, tail_token_type_ids,
                triples_input_ids, triples_mask, triples_token_type_ids,
                **kwargs):
        bs = hr_input_ids.size(0)
        h, r = self.hr_encode(hr_input_ids, hr_mask, hr_token_type_ids)
        related_triples = self.triples_encode(triples_input_ids, triples_mask, triples_token_type_ids)
        t = self.tail_encode(tail_input_ids, tail_mask, tail_token_type_ids)
        label = torch.arange(bs).to(h.device)
        embed_dict = {
            'head' : h,
            'relation' : r,
            'related_triples' : related_triples,
            'tail' : t,
            'label' : label,
        }
        return embed_dict

    def compute_logits(self, embed_dict, mode='train'):
        head, relation, tail = embed_dict['head'], embed_dict['relation'], embed_dict['tail']
        related_triples = embed_dict['related_triples']
        fixed_head = torch.mm(related_triples, self.W) + head
        hr = fixed_head * relation
        logits = hr.mm(tail.t())
        if mode == 'train':
            logits -= torch.zeros_like(logits).fill_diagonal_(self.args.margin)
        logits *= self.alpha.exp()
        return logits
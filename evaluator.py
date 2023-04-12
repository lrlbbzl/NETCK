import torch
from torch import nn
import os
from collections import OrderedDict
from model import NETCK
from logger_config import logger
from config import args
from structure import EntityDict, EntityItem
from dataset import Triplet, collate, Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

class AttributeDict():
    pass

class Evaluator():
    def __init__(self, model_path):
        self.train_args = AttributeDict()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.load(model_path)

    def load(self, model_path):
        assert os.path.exists(model_path), 'Model path doesn\'t exist.'
        ckt_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.train_args.__dict__ = ckt_dict['args']
        for k, v in args.__dict__.items():
            if k not in self.train_args.__dict__:
                self.train_args.__dict__[k] = v
        # create model
        self.model = NETCK(self.train_args)
        state_dict = ckt_dict['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[len('module.'):]
            new_state_dict[k] = v
        self.model.load_state_dict(new_state_dict, strict=True)
        self.model.eval()

        if self.device == 'cuda':
            self.model.cuda()
        logger.info('Load model successfully from {}.'.format(model_path))

    def entity_embedding(self, ):
        all_entities = EntityDict(self.train_args.entity_path)
        logger.info('Load {} entities from {}.'.format(len(all_entities), self.train_args.entity_path))
        exs = []
        for ent in all_entities.entities:
            exs.append(Triplet(tail_id=ent.entity_id)) # only used for generate entity embedding
        entity_data = Dataset(path_list='', data=exs)
        data_loader = DataLoader(
            dataset=entity_data,
            num_workers=self.train_args.loader_workers,
            batch_size=self.train_args.batch_size, 
            collate_fn=collate,
            shuffle=False
        )

        entities_embedding = []
        for i, batch_data in enumerate(tqdm(data_loader)):
            if self.device == 'cuda':
                batch_data = batch_data.cuda()
            embed = self.model.tail_encode(batch_data['tail_input_ids'], batch_data['tail_mask'], batch_data['tail_token_type_ids'])
            entities_embedding.append(embed)
        entities_embedding = torch.cat(entities_embedding, dim=0) # (num_ent, hidden_size)
        return entities_embedding

    def embed_generator(self, data):
        dataset = Dataset(path_list='', data=data)
        dataloader = DataLoader(
            dataset=dataset, 
            batch_size=self.train_args.batch_size, 
            num_workers=self.train_args.loader_workers, 
            collate_fn=collate,
            shuffle=False)
        h_embed, r_embed = [], []
        triples_embed = []
        for i, batch_data in enumerate(tqdm(dataloader)):
            if self.device == 'cuda':
                batch_data = batch_data.cuda()
            temp_h, temp_r = self.model.hr_encode(batch_data['hr_input_ids'], batch_data['hr_mask'], batch_data['hr_token_type_ids'])
            temp_triples = self.model.triples_encode(batch_data['triples_input_ids'], batch_data['triples_mask'], batch_data['triples_token_type_ids'])
            h_embed.append(temp_h)
            r_embed.append(temp_r)
            triples_embed.append(temp_triples)
        head_embedding, relation_embedding = torch.cat(h_embed, dim=0), torch.cat(r_embed, dim=0)
        triples_embedding = torch.cat(triples_embed, dim=0)
        return head_embedding, relation_embedding, triples_embedding
        


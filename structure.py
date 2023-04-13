import torch
import numpy as np
import json
from tqdm import tqdm
from collections import deque
from dataclasses import dataclass
from logger_config import logger

@dataclass
class EntityItem:
    entity_id: str
    entity: str
    entity_desc: str = ''

class EntityDict(object):
    def __init__(self, entity_path: str):
        entity = json.load(open(entity_path, 'r', encoding='utf-8'))
        self.entities = [EntityItem(**x) for x in entity]
        self.id2idx = {x.entity_id : i for i, x in enumerate(self.entities)}
        self.id2entity = {x.entity_id : x for x in self.entities}
        logger.info('Load {} entities from {}'.format(len(self.entities), entity_path))
    

class TotalGraph(object):
    def __init__(self, train_path: str):
        logger.info('Build total graph from {}'.format(train_path))
        self.train_data = json.load(open(train_path, 'r', encoding='utf-8'))
        self.totalGraph = dict()
        for x in tqdm(self.train_data):
            h_id, t_id = x['head_id'], x['tail_id']
            if h_id not in self.totalGraph:
                self.totalGraph.update({h_id : set()})
            if t_id not in self.totalGraph:
                self.totalGraph.update({t_id : set()})
            self.totalGraph[h_id].add(t_id)
            self.totalGraph[t_id].add(h_id)
        logger.info('Build done with {} nodes in graph.'.format(len(self.totalGraph)))

    def get_neighbors_ids(self, entity_id: str, max_nodes=10):
        neighbors_ids = self.totalGraph.get(entity_id, set())
        return sorted(list(neighbors_ids))[:max_nodes]
    
    def get_nhop_neighbors(self, entity_id: str, nhop: int, id_to_num: dict):
        seen_entity = set() # entities which has been seen in exvacating
        neighbors = dict() # degree : neighbors
        if nhop <= 0:
            return neighbors
        q = deque([entity_id])
        for i in range(1, nhop + 1):
            length = len(q)
            neighbors.update({i : set()})
            for idx in range(length):
                x = q.popleft()
                node_neighbors = self.totalGraph.get(x, set())
                for neighbor in node_neighbors:
                    if neighbor not in seen_entity:
                        seen_entity.add(neighbor)
                        q.append(neighbor)
                        neighbors[i].add(neighbor)
        return neighbors
    
    def get_related_triples(self, ):
        related_triples = dict()
        logger.info('Generate related triples for each entity.')
        for x in tqdm(self.train_data):
            head_id, relation, tail_id = x['head_id'], x['relation'], x['tail_id']
            if tail_id not in related_triples:
                related_triples.update({tail_id : []})
            related_triples[tail_id].append((head_id, relation))
        logger.info('Generation done. {} entities have related triples.'.format(len(related_triples)))
        return related_triples

if __name__ == '__main__':
    xx = TotalGraph('./data/FB15k-237/train.txt.json')
    yy = EntityDict('./data/FB15k-237/entities.json')
    temp = xx.get_related_triples()
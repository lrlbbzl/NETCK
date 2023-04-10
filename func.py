import torch
import numpy as np
from mydataset import Triplet
from typing import Optional, List
from copy import deepcopy
    

# def generate_negative_mask(triplet: List[Triplet], entityDict):
#     """
#     function: mask those false negative samples which appear given head and relation
#     """
#     row_entity_ids = torch.LongTensor([entityDict.id2idx[x.tail_id] for x in triplet])
#     col_entity_ids = deepcopy(row_entity_ids)
#     mask = (row_entity_ids.unsqueeze(1) != col_entity_ids.unsqueeze(0)).fill_diagonal_(True)

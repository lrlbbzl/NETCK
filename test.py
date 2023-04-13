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
print(len(entityDict.id2entity))
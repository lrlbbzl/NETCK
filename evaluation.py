import torch
from evaluator import Evaluator
from config import args
from dataset import Dataset
from torch.utils.data import DataLoader
from structure import EntityDict
from tqdm import tqdm
from logger_config import logger
import json
import os
from time import time
import pickle
from dataclasses import dataclass, asdict

entityDict = EntityDict(args.entity_path)

@dataclass
class PredInfo:
    head: str
    relation: str
    tail: str
    pred_tail: str
    pred_score: float
    topk_score_info: str
    rank: int
    correct: bool


@torch.no_grad()
def compute_metrics(head_tensor: torch.tensor,
                    relation_tensor: torch.tensor,
                    triples_tensor: torch.tensor,
                    entities_tensor: torch.tensor,
                    target,
                    tester: Evaluator,
                    k=3, batch_size=256):
    entity_cnt = len(entityDict.id2idx)
    assert entity_cnt == entities_tensor.size(0)
    target = torch.LongTensor(target).unsqueeze(-1).to(head_tensor.device)
    total = head_tensor.size(0)
    topk_scores, topk_indices = [], []
    ranks = []

    mean_rank, mrr, hit1, hit3, hit10 = 0, 0, 0, 0, 0
    for start in tqdm.tqdm(range(0, total, batch_size)):
        end = start + batch_size
        # batch_size * entity_cnt
        fixed_head = torch.mm(triples_tensor[start:end, :], tester.model.W) + head_tensor[start:end, :]
        hr = fixed_head * relation_tensor[start:end, :]
        logits = hr.mm(entities_tensor.t())
        assert entity_cnt == logits.size(1)
        batch_target = target[start:end]

        # re-ranking based on topological structure
        # rerank_by_graph(logits, examples[start:end], entity_dict=entity_dict)    

        # filter known triplets
        # for idx in range(logits.size(0)):
        #     mask_indices = []
        #     cur_ex = examples[start + idx]
        #     gold_neighbor_ids = all_triplet_dict.get_neighbors(cur_ex.head_id, cur_ex.relation)
        #     if len(gold_neighbor_ids) > 10000:
        #         logger.debug('{} - {} has {} neighbors'.format(cur_ex.head_id, cur_ex.relation, len(gold_neighbor_ids)))
        #     for e_id in gold_neighbor_ids:
        #         if e_id == cur_ex.tail_id:
        #             continue
        #         mask_indices.append(entity_dict.entity_to_idx(e_id))
        #     mask_indices = torch.LongTensor(mask_indices).to(logits.device)
        #     logits[idx].index_fill_(0, mask_indices, -1)

        batch_sorted_score, batch_sorted_indices = torch.sort(logits, dim=-1, descending=True)
        target_rank = torch.nonzero(batch_sorted_indices.eq(batch_target).long(), as_tuple=False)
        assert target_rank.size(0) == logits.size(0)
        for idx in range(logits.size(0)):
            idx_rank = target_rank[idx].tolist()
            assert idx_rank[0] == idx
            cur_rank = idx_rank[1]

            # 0-based -> 1-based
            cur_rank += 1
            mean_rank += cur_rank
            mrr += 1.0 / cur_rank
            hit1 += 1 if cur_rank <= 1 else 0
            hit3 += 1 if cur_rank <= 3 else 0
            hit10 += 1 if cur_rank <= 10 else 0
            ranks.append(cur_rank)

        topk_scores.extend(batch_sorted_score[:, :k].tolist())
        topk_indices.extend(batch_sorted_indices[:, :k].tolist())

    metrics = {'mean_rank': mean_rank, 'mrr': mrr, 'hit@1': hit1, 'hit@3': hit3, 'hit@10': hit10}
    metrics = {k: round(v / total, 4) for k, v in metrics.items()}
    assert len(topk_scores) == total
    return topk_scores, topk_indices, metrics, ranks

def eval_single_direction(tester: Evaluator, entity_embedding, data, eval_forward: bool):
    start_time = time()
    head_embedding, relation_embedding, triples_embedding = tester.embed_generator(data)
    target = [x.tail_id for x in data]
    target = [entityDict.id2idx[x] for x in target]
    topk_scores, topk_indices, metrics, ranks = compute_metrics(
        head_tensor=head_embedding, relation_tensor=relation_embedding, triples_tensor=triples_embedding,
        entities_tensor=entity_embedding, target=target, batch_size=args.batch_size
    )
    eval_dir = 'forward' if eval_forward else 'backward'
    logger.info('{} metrics: {}'.format(eval_dir, json.dumps(metrics)))

    pred_infos = []
    for idx, ex in enumerate(data):
        cur_topk_scores = topk_scores[idx]
        cur_topk_indices = topk_indices[idx]
        pred_idx = cur_topk_indices[0]
        cur_score_info = {entityDict.entities[topk_idx].entity: round(topk_score, 3)
                          for topk_score, topk_idx in zip(cur_topk_scores, cur_topk_indices)}

        pred_info = PredInfo(head=ex.head, relation=ex.relation,
                             tail=ex.tail, pred_tail=entityDict.entities[pred_idx].entity,
                             pred_score=round(cur_topk_scores[0], 4),
                             topk_score_info=json.dumps(cur_score_info),
                             rank=ranks[idx],
                             correct=pred_idx == target[idx])
        pred_infos.append(pred_info)

    prefix, basename = os.path.dirname(args.model_path), os.path.basename(args.model_path)
    split = os.path.basename(args.test_path)
    with open('{}/eval_{}_{}_{}.json'.format(prefix, split, eval_dir, basename), 'w', encoding='utf-8') as writer:
        writer.write(json.dumps([asdict(info) for info in pred_infos], ensure_ascii=False, indent=4))

    logger.info('Evaluation takes {} seconds'.format(round(time() - start_time, 3)))
    return metrics

def eval_on_test():
    tester = Evaluator(args.model_path)
    entity_embedding = tester.entity_embedding()
    sample_data = Dataset(args.test_path)
    forward_data = sample_data.load_data(path=args.test_path, add_forward=True, add_reciprocal=False)
    reciprocal_data = sample_data.load_data(path=args.test_path, add_forward=False, add_reciprocal=True)
    forward_metrics = eval_single_direction(tester,
                                            entity_embedding,
                                            forward_data,
                                            eval_forward=True)
    backward_metrics = eval_single_direction(tester,
                                            entity_embedding,
                                            reciprocal_data,
                                            eval_forward=False)
    metrics = {k: round((forward_metrics[k] + backward_metrics[k]) / 2, 4) for k in forward_metrics}
    logger.info('Averaged metrics: {}'.format(metrics))

    prefix, basename = os.path.dirname(args.eval_model_path), os.path.basename(args.eval_model_path)
    split = os.path.basename(args.valid_path)
    with open('{}/metrics_{}_{}.json'.format(prefix, split, basename), 'w', encoding='utf-8') as writer:
        writer.write('forward metrics: {}\n'.format(json.dumps(forward_metrics)))
        writer.write('backward metrics: {}\n'.format(json.dumps(backward_metrics)))
        writer.write('average metrics: {}\n'.format(json.dumps(metrics)))

if __name__ == '__main__':
    eval_on_test()

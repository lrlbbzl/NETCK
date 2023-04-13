import torch
from torch import nn
from torch.utils.data import DataLoader
import warnings 
warnings.filterwarnings("ignore")
from logger_config import logger
from model import NETCK
from dataset import Dataset, collate
from utils import AverageMeter, ProgressMeter, accuracy, move_to_cuda

import shutil
import json
import os
import pickle
import math
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, AdamW

class Trainer():
    def __init__(self, args):
        self.args = args
        logger.info('Creating Model.')
        self.model = NETCK(args)
        self.best_metric = 0.0
        # logger.info(self.model)
        # cuda setting
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model).cuda()
        elif torch.cuda.is_available():
            logger.info('Use cuda.')
            self.model.cuda()
        else:
            logger.info('No available gpus, run on cpu.')
        # optimization
        self.optimizer = AdamW([param for param in self.model.parameters() if param.requires_grad], lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.criterion = nn.CrossEntropyLoss().cuda()

        # data
        self.train_data = Dataset([self.args.train_path])
        self.valid_data = Dataset([self.args.valid_path])
        self.train_dataloader = DataLoader(
            self.train_data,
            batch_size=self.args.batch_size,
            collate_fn=collate,
            num_workers=self.args.loader_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=True
        )
        self.valid_dataloader = DataLoader(
            self.valid_data,
            batch_size=self.args.batch_size,
            collate_fn=collate,
            num_workers=self.args.loader_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=True
        )

        total_step = math.ceil(self.args.epoch * len(self.train_data) / self.args.batch_size)
        self.args.warmup_step = min(self.args.warmup_step, total_step // 10)
        logger.info('Total step in training is {}, warm up to {}.'.format(total_step, self.args.warmup_step))
        self.scheduler = self.create_lr_scheduler(self.args.warmup_step)
        if self.args.use_amp:
            self.scaler = torch.cuda.amp.scaler()
    
    def create_lr_scheduler(self, num_training_steps):
        if self.args.lr_scheduler == 'linear':
            return get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                   num_warmup_steps=self.args.warmup_step,
                                                   num_training_steps=num_training_steps)
        elif self.args.lr_scheduler == 'cosine':
            return get_cosine_schedule_with_warmup(optimizer=self.optimizer,
                                                   num_warmup_steps=self.args.warmup_step,
                                                   num_training_steps=num_training_steps)
        else:
            assert False, 'Unknown lr scheduler: {}'.format(self.args.lr_scheduler)

    def train_epoch(self, epoch):
        # metrics logger
        losses = AverageMeter('Loss', ':.4')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top3 = AverageMeter('Acc@3', ':6.2f')
        progress = ProgressMeter(
            len(self.train_dataloader),
            [losses, top1, top3],
            prefix="Epoch: [{}]".format(epoch))
        
        # training
        for i, batch_data in enumerate(self.train_dataloader):
            self.model.train()
            if self.device == 'cuda':
                batch_data = move_to_cuda(batch_data)
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**batch_data)
            else:
                outputs = self.model(**batch_data)
            logits = self.model.compute_logits(outputs, mode='train')
            bs = logits.size(0)
            labels = outputs['label']
            loss = self.criterion(logits, labels)
            loss += self.criterion(logits[:bs].t(), labels)
            losses.update(loss.item(), bs)

            acc1, acc3 = accuracy(logits, labels, topk=(1, 3))
            top1.update(acc1.item(), bs)
            top3.update(acc3.item(), bs)

            self.optimizer.zero_grad()
            if self.args.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            self.scheduler.step()

            if (i + 1) % self.args.print_freq == 0:
                progress.display(i + 1)
        
        if (epoch + 1) % self.args.eval_freq == 0:
            metric_dict = self.evaluate(epoch)
            acc1 = metric_dict['Acc@1']
            is_best = False
            if acc1 > self.best_metric:
                self.best_metric = acc1
                is_best = True
            self.save_model(epoch, is_best)
        logger.info('Learning rate: {}'.format(self.scheduler.get_last_lr()[0]))

    @torch.no_grad()
    def evaluate(self, epoch):
        losses = AverageMeter('Loss', ':.4')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top3 = AverageMeter('Acc@3', ':6.2f')

        self.model.eval()
        for i, batch_data in enumerate(self.valid_dataloader):
            if self.device == 'cuda':
                batch_data = move_to_cuda(batch_data)
            bs = len(batch_data)
            outputs = self.model(**batch_data)
            logits = self.model.compute_logits(outputs, mode='valid')
            labels = outputs['label']
            loss = self.criterion(logits, labels)

            losses.update(loss.item(), bs)

            acc1, acc3 = accuracy(logits, labels, topk=(1, 3))
            top1.update(acc1.item(), bs)
            top3.update(acc3.item(), bs)

        metric_dict = {'Acc@1': round(top1.avg, 3),
                       'Acc@3': round(top3.avg, 3),
                       'loss': round(losses.avg, 3)}
        logger.info('Epoch {}, valid metric: {}'.format(epoch, json.dumps(metric_dict)))
        return metric_dict
    
    def save_model(self, epoch, is_best:bool):
        file_name = '{}/checkpoint_{}.mdl'.format(self.args.save_dir, epoch)
        save_dict = {
            'epoch' : epoch,
            'args' : self.args.__dict__,
            'state_dict' : self.model.state_dict(),
        }
        torch.save(save_dict, file_name)
        if is_best:
            shutil.copyfile(file_name, os.path.join(self.args.save_dir, 'best_model.mdl'))
        return None

    def train_loop(self, ):
        for i in range(self.args.epoch):
            self.train_epoch(i)

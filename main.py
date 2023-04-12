import torch
from trainer import Trainer
from config import args

if __name__ == '__main__':
    trainer = Trainer(args=args)
    trainer.train_loop()
    
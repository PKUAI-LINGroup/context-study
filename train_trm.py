# -*- coding: utf-8 -*-
# @File        : train_trm.py
# @Description : Run this script to train Transformer-based dialog model.


import os
import argparse

import torch

from trainers.trm_trainer import TrmTrainer
from vocabs.gpt2_tokenizer import GPT2Vocab
from datasets.trm_dataset import TrmDataset

from models.trm_model import TrmModel
from utils import setup_seed, TrmModelConfig, print_args


def main(args):
    setup_seed(args.seed)
    device = torch.device(args.device)
    print_args(args)

    gpt2_vocab = GPT2Vocab(model_path=args.gpt2_vocab_dir)

    print('loading train dataset:')
    train_dataset = TrmDataset(cache_data_path=args.train_dataset, vocab=gpt2_vocab,
                               max_history_utterance=args.max_history_utterance,
                               max_seq_len=args.max_seq_len)
    print(f'train dataset has {len(train_dataset)} samples')

    print('loading valid dataset:')
    valid_dataset = TrmDataset(cache_data_path=args.valid_dataset, vocab=gpt2_vocab,
                               max_history_utterance=args.max_history_utterance,
                               max_seq_len=args.max_seq_len)
    print(f'valid dataset has {len(valid_dataset)} samples')

    # initialize model
    model_config = TrmModelConfig(vocab_size=len(gpt2_vocab))

    model = TrmModel(model_config=model_config, vocab=gpt2_vocab, args=args)

    trainer = TrmTrainer(model=model,
                         args=args,
                         train_dataset=train_dataset,
                         valid_dataset=valid_dataset,
                         device=device,
                         vocab=gpt2_vocab
                         )

    # load checkpoint
    last_epoch = 0
    trainer.train(last_epoch=last_epoch)


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt2_vocab_dir", default="./gpt2_vocab", help="path to GPT2 tokenizer vocab file")
    parser.add_argument("--train_dataset", default="./processed_data/dailydialog/train.cache", type=str,
                        help="processed train dataset path")
    parser.add_argument("--valid_dataset", default="./processed_data/dailydialog/valid.cache", type=str,
                        help="processed valid dataset path")
    parser.add_argument("--max_history_utterance", type=int, default=10,
                        help="how many history utterance to keep (including the last utterance)")
    parser.add_argument("--max_seq_len", type=int, default=50, help="max length for each utterance")
    parser.add_argument("--seed", default=0, type=int, help="random seed")
    parser.add_argument("--device", default='cuda' if torch.cuda.is_available() else 'cpu', type=str,
                        help="cpu or cuda")
    parser.add_argument("--n_epochs", default=1, type=int, help="number of training epochs")
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--gradient_accumulate_steps", default=1, type=int, help="accumulate gradient on several steps")
    parser.add_argument("--lr", default=6.25e-5, type=float, help="learning rate")
    parser.add_argument("--clip_grad", type=float, default=1.0, help="clip gradient threshold")
    parser.add_argument("--save_model_dir", default="./checkpoints/trm/tmp", help="path to save model checkpoints")
    parser.add_argument("--log_dir", type=str, default="./logs/trm/tmp", help="path to tensorboard log")
    parser.add_argument("--save_interval", type=int, default=1)

    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()

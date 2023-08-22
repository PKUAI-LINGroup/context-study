# -*- coding: utf-8 -*-
# @File        : train_gpt2.py
# @Description : Run this script to train GPT2-based dialog model.


import os
import argparse

import torch

from trainers.gpt2_trainer import GPT2Trainer
from vocabs.gpt2_tokenizer import GPT2Vocab
from datasets.gpt2_dataset import GPT2Dataset

from models.gpt2_model import GPT2Model
from utils import setup_seed, print_args
from modules.custom_gpt2_module import CustomGPT2Module


def main(args):
    setup_seed(args.seed)
    device = torch.device(args.device)
    print_args(args)

    gpt2_vocab = GPT2Vocab(model_path=args.gpt2_vocab_dir)

    print('loading train dataset:')
    train_dataset = GPT2Dataset(cache_data_path=args.train_dataset, vocab=gpt2_vocab,
                                max_history_utterance=args.max_history_utterance,
                                max_seq_len=args.max_seq_len)
    print(f'train dataset has {len(train_dataset)} samples')

    print('loading valid dataset:')
    valid_dataset = GPT2Dataset(cache_data_path=args.valid_dataset, vocab=gpt2_vocab,
                                max_history_utterance=args.max_history_utterance,
                                max_seq_len=args.max_seq_len)
    print(f'valid dataset has {len(valid_dataset)} samples')

    # initialize model
    gpt2_module = CustomGPT2Module.from_pretrained(args.gpt2_model_dir)
    gpt2_module.resize_token_embeddings(new_num_tokens=len(gpt2_vocab))
    gpt2_module.set_utterance_position_embeddings(n_utterance_positions=args.max_history_utterance + 1,
                                                  n_embd=gpt2_module.config.n_embd)

    model = GPT2Model(core_module=gpt2_module, vocab=gpt2_vocab)

    trainer = GPT2Trainer(model=model,
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
    parser.add_argument("--gpt2_model_dir", default="/gpt2_model", help="path to GPT2 pretrained model file")
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
    parser.add_argument("--save_model_dir", default="./checkpoints/gpt2/tmp", help="path to save model checkpoints")
    parser.add_argument("--log_dir", type=str, default="./logs/gpt2/tmp", help="path to tensorboard log")
    parser.add_argument("--save_interval", type=int, default=1)

    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()

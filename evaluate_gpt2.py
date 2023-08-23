# -*- coding: utf-8 -*-
# @File        : evaluate_gpt2.py
# @Description :


import os
import argparse

import torch

from evaluators.gpt2_evaluator import GPT2Evaluator
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
    print('loading test dataset:')
    test_dataset = GPT2Dataset(cache_data_path=args.test_dataset, vocab=gpt2_vocab,
                               max_history_utterance=args.max_history_utterance,
                               max_seq_len=args.max_seq_len)
    print(f'test dataset has {len(test_dataset)} samples')

    # initialize model
    gpt2_module = CustomGPT2Module.from_pretrained(args.gpt2_model_dir)
    gpt2_module.resize_token_embeddings(new_num_tokens=len(gpt2_vocab))
    gpt2_module.set_utterance_position_embeddings(n_utterance_positions=args.max_history_utterance + 1,
                                                  n_embd=gpt2_module.config.n_embd)

    model = GPT2Model(core_module=gpt2_module, vocab=gpt2_vocab)

    evaluator = GPT2Evaluator(model=model,
                              args=args,
                              test_dataset=test_dataset,
                              device=device,
                              vocab=gpt2_vocab
                              )
    evaluator.evaluate()


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt2_vocab_dir", default="./gpt2_vocab", help="path to GPT2 tokenizer vocab file")
    parser.add_argument("--gpt2_model_dir", default="./gpt2_model", help="path to GPT2 pretrained model file")
    parser.add_argument("--test_dataset", default="./processed_data/dailydialog/test.cache", type=str,
                        help="processed test dataset path")
    parser.add_argument("--max_history_utterance", type=int, default=10,
                        help="how many history utterance to keep (including the last utterance)")
    parser.add_argument("--max_seq_len", type=int, default=50, help="max length for each utterance")
    parser.add_argument("--seed", default=0, type=int, help="random seed")
    parser.add_argument("--device", default='cuda' if torch.cuda.is_available() else 'cpu', type=str,
                        help="cpu or cuda")
    parser.add_argument("--checkpoint_path", default="./checkpoints/gpt2/tmp/checkpoint1.pt",
                        help="path to load model checkpoint")
    parser.add_argument("--max_predict_len", type=int, default=50, help="max predicted response sequence length")
    parser.add_argument("--save_result_path", default="./results/tmp.jsonl", help="path to save prediction results")

    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()

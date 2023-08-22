# -*- coding: utf-8 -*-
# @File        : utils.py


import os
import random
import numpy as np

import torch

from transformers import GPT2Config


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False


class TrmModelConfig:
    def __init__(self, vocab_size):
        self.encoder_config = GPT2Config(vocab_size=vocab_size, n_embd=256, n_layer=3, n_head=2)
        self.decoder_config = GPT2Config(vocab_size=vocab_size, n_embd=256, n_layer=3, n_head=2)


def print_args(args):
    for key in args.__dict__:
        print(f"{key}: {args.__dict__[key]}")

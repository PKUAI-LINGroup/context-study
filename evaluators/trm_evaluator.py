# -*- coding: utf-8 -*-
# @File        : trm_evaluator.py
# @Description :


import os
import math
import jsonlines
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from evaluators.evaluate_utils import greedy_search_for_trm


class TrmEvaluator:
    def __init__(self, model, args, test_dataset, device, vocab):
        self.args = args
        self.device = device
        self.model = model.to(device)
        # load checkpoint
        self.load_state_dict(torch.load(args.checkpoint_path, map_location=self.device))
        print('Weights loaded from {}'.format(args.checkpoint_path))
        self.vocab = vocab
        self.test_dataset = test_dataset

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict['model'], strict=False)

    def evaluate(self):
        self.model.eval()
        test_dataloader = DataLoader(self.test_dataset, batch_size=1,
                                     collate_fn=self.test_dataset.collate_func, num_workers=4)
        tqdm_data = tqdm(test_dataloader, desc='Test: ')
        test_losses = []
        with torch.no_grad():
            with jsonlines.open(self.args.save_result_path, "w") as fout:
                for i, data in enumerate(tqdm_data):
                    data = {key: data[key].to(self.device) for key in data}
                    model_out = self.model(context=data["context"],
                                           context_type_ids=data["context_type_ids"],
                                           context_position_ids=data["context_position_ids"],
                                           context_utterance_position_ids=data["context_utterance_position_ids"],
                                           tgt=data["tgt"],
                                           tgt_type_ids=data["tgt_type_ids"],
                                           tgt_position_ids=data["tgt_position_ids"],
                                           tgt_utterance_position_ids=data["tgt_utterance_position_ids"],
                                           lm_labels=data["tgt"]
                                           )
                    ppl = torch.exp(model_out["lm_loss"]).item()
                    test_losses.append(model_out["lm_loss"].item())

                    predict_ids = greedy_search_for_trm(
                        context=data["context"],
                        context_type_ids=data["context_type_ids"],
                        context_position_ids=data["context_position_ids"],
                        context_utterance_position_ids=data["context_utterance_position_ids"],
                        max_len=self.args.max_predict_len,
                        vocab=self.vocab,
                        model=self.model
                    )

                    # context
                    context = self.test_dataset.contexts["context"][i]
                    context_strings = []
                    for sent_idx, sent in enumerate(context):
                        sent_string = self.vocab.ids2string(sent, skip_special_tokens=True,
                                                            clean_up_tokenization_spaces=False)
                        context_strings.append(sent_string)
                    # tgt
                    ref_sent = self.vocab.ids2string(data["tgt"][0],
                                                     skip_special_tokens=True,
                                                     clean_up_tokenization_spaces=False)

                    # predict
                    pred_sent = self.vocab.ids2string(predict_ids[0],
                                                      skip_special_tokens=True,
                                                      clean_up_tokenization_spaces=False)

                    fout.write({
                        "sample_idx": i + 1,
                        "context": context_strings,
                        "tgt": ref_sent,
                        "predict": pred_sent,
                        "loss": model_out["lm_loss"].item(),
                        "ppl": ppl
                    })

        ave_loss = float(np.mean(test_losses))

        print(f"test ppl: {torch.exp(torch.tensor(ave_loss)).item()}")

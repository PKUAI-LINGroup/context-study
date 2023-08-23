# -*- coding: utf-8 -*-
# @File        : gpt2_evaluator.py
# @Description :


import os
import math
import jsonlines
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from evaluators.evaluate_utils import greedy_search_for_gpt2


class GPT2Evaluator:
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
                    model_out = self.model(input_ids=data["input_ids"],
                                           type_ids=data["type_ids"],
                                           position_ids=data["position_ids"],
                                           utterance_position_ids=data["utterance_position_ids"],
                                           lm_labels=data["lm_labels"]
                                           )
                    ppl = torch.exp(model_out["lm_loss"]).item()
                    test_losses.append(model_out["lm_loss"].item())

                    dialog = self.test_dataset.dialogs[i]
                    data_for_generate = self.test_dataset.build_data(
                        context=dialog["context"], vocab=self.vocab,
                        max_history_utterance=self.args.max_history_utterance, max_seq_len=self.args.max_seq_len)
                    predict_ids = greedy_search_for_gpt2(
                        input_ids=torch.tensor(data_for_generate["input_ids"], dtype=torch.long).to(
                            self.device).unsqueeze(0),
                        type_ids=torch.tensor(data_for_generate["type_ids"], dtype=torch.long).to(
                            self.device).unsqueeze(0),
                        position_ids=torch.tensor(data_for_generate["position_ids"], dtype=torch.long).to(
                            self.device).unsqueeze(0),
                        utterance_position_ids=torch.tensor(data_for_generate["utterance_position_ids"],
                                                            dtype=torch.long).to(self.device).unsqueeze(0),
                        max_len=self.args.max_predict_len,
                        vocab=self.vocab,
                        model=self.model
                    )

                    # context
                    context = dialog["context"]
                    context_strings = []
                    for sent_idx, sent in enumerate(context):
                        sent_string = self.vocab.ids2string(sent, skip_special_tokens=True,
                                                            clean_up_tokenization_spaces=False)
                        context_strings.append(sent_string)
                    # tgt
                    ref_sent = self.vocab.ids2string(dialog["tgt"],
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

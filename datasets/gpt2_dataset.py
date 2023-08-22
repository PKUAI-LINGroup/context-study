# -*- coding: utf-8 -*-
# @File        : gpt2_dataset.py
# @Description :


"""
suppose context is (x_1, y_1, ..., x_t), tgt is y_t
+ GPT2 Input:
    + Sequence:           x_1         <eos>   y_1       <eos> ... x_t         <eos>   <bos> y_t <eos>
    + Type:               <human> ... <human> <bot> ... <bot> ... <human> ... <human> <bot> ... <bot>
    + Token Position:     0 1 ...             0 1 ...         ... 0 1 ...             0 1 ...
    + Utterance Position: 2t-1 ...            2t-2 ...        ... 1 ...               0 ...
"""


import logging
from itertools import chain

import torch
from torch.utils.data import Dataset


class GPT2Dataset(Dataset):
    def __init__(self, cache_data_path, vocab, max_history_utterance, max_seq_len):
        """
        :param max_history_utterance: how many history utterance to keep (including the last utterance)
        :param max_seq_len: max length for each utterance sequence
        """
        super(GPT2Dataset, self).__init__()
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__file__)
        self.vocab = vocab
        dialogs = torch.load(cache_data_path)
        self.dialogs = dialogs
        self.contexts = {"context": []}
        self.logger.info("building data from segments")
        data = [self.build_data(context=dialog["context"], tgt=dialog["tgt"], vocab=vocab,
                                max_history_utterance=max_history_utterance, max_seq_len=max_seq_len)
                for dialog in dialogs]
        self.logger.info("padding and converting to tensor")
        self.pad_data = self.get_padding_data(data)

    def __len__(self):
        return self.pad_data["input_ids"].shape[0]

    def __getitem__(self, item):
        return {"input_ids": self.pad_data["input_ids"][item, :],
                "type_ids": self.pad_data["type_ids"][item, :],
                "position_ids": self.pad_data["position_ids"][item, :],
                "utterance_position_ids": self.pad_data["utterance_position_ids"][item, :],
                "lm_labels": self.pad_data["lm_labels"][item, :]
                }

    def build_data(self, context, vocab, max_history_utterance, max_seq_len, tgt=None):
        self.contexts["context"].append(context)
        context = context[-max_history_utterance:]
        context = [seq[:max_seq_len - 1] + [vocab.eos_id] for seq in context]
        utterance_position_ids = [len(context) - utter_idx for utter_idx, seq in enumerate(context) for _ in seq]
        type_ids = [vocab.human_id if (len(context) - i) % 2 else vocab.bot_id
                    for i, s in enumerate(context) for _ in s]
        position_ids = [i for s in context for i in range(len(s))]
        input_ids = list(chain(*context))
        if tgt is not None:
            tgt = [vocab.bos_id] + tgt[: max_seq_len - 2] + [vocab.eos_id]
            type_ids = type_ids + [vocab.bot_id] * len(tgt)
            position_ids = position_ids + list(range(len(tgt)))
            utterance_position_ids = utterance_position_ids + [0 for _ in tgt]
            lm_labels = [vocab.pad_id] * (len(input_ids) + 1) + tgt[1:]
            input_ids += tgt
        else:
            lm_labels = None

        return {"input_ids": input_ids,
                "type_ids": type_ids,
                "position_ids": position_ids,
                "utterance_position_ids": utterance_position_ids,
                "lm_labels": lm_labels}

    def get_padding_data(self, data):
        pad_data = {"input_ids": [],  # n_samples, max_total_len
                    "type_ids": [],  # n_samples, max_total_len
                    "position_ids": [],  # n_samples, max_total_len
                    "utterance_position_ids": [],  # n_samples, max_total_len
                    "lm_labels": [],  # n_samples, max_total_len
                    }
        for instance in data:
            for key_name in instance:
                pad_data[key_name].append(instance[key_name])

        max_total_len = max(len(sequence) for sequence in pad_data["input_ids"])
        for key in ["input_ids", "type_ids", "lm_labels"]:
            pad_data[key] = self.pad_and_convert_to_tensor(pad_data[key], pad_id=self.vocab.pad_id,
                                                           max_seq_len=max_total_len)
        for key in ["position_ids", "utterance_position_ids"]:
            pad_data[key] = self.pad_and_convert_to_tensor(pad_data[key], pad_id=0, max_seq_len=max_total_len)

        return {
            "input_ids": pad_data["input_ids"],
            "type_ids": pad_data["type_ids"],
            "position_ids": pad_data["position_ids"],
            "utterance_position_ids": pad_data["utterance_position_ids"],
            "lm_labels": pad_data["lm_labels"]
        }

    def collate_func(self, instances):
        batch_data = {}
        for key in instances[0]:
            batch_data[key] = torch.stack([instance[key] for instance in instances])
        return batch_data

    @staticmethod
    def pad_and_convert_to_tensor(sequences, pad_id, max_seq_len=None):
        if max_seq_len is None:
            max_seq_len = max(len(sequence) for sequence in sequences)
        tensor_data = [seq + [pad_id] * (max_seq_len - len(seq)) for seq in sequences]
        tensor_data = torch.tensor(tensor_data, dtype=torch.long)
        return tensor_data

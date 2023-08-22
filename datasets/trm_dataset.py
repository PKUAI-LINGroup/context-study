# -*- coding: utf-8 -*-
# @File        : trm_dataset.py
# @Description :


"""
suppose context is (x_1, y_1, ..., x_t), tgt is y_t
+ Encoder:
    + Sequence:           x_1         <eos>   y_1       <eos> ... x_t         <eos>
    + Type:               <human> ... <human> <bot> ... <bot> ... <human> ... <human>
    + Token Position:     0 1 ...             0 1 ...         ... 0 1 ...
    + Utterance Position: 2t-1 ...            2t-2 ...        ... 1 ...
+ Decoder:
    + Sequence:           <bos> y_t <eos>
    + Type                <bot> ... <bot>
    + Token Position:     0 1 ...
    + Utterance Position: 0 ...
"""

import logging
from itertools import chain

import torch
from torch.utils.data import Dataset


class TrmDataset(Dataset):
    def __init__(self, cache_data_path, vocab, max_history_utterance, max_seq_len):
        """
        :param max_history_utterance: how many history utterance to keep (including the last utterance)
        :param max_seq_len: max length for each utterance sequence
        """
        super(TrmDataset, self).__init__()
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__file__)
        self.vocab = vocab
        dialogs = torch.load(cache_data_path)
        self.dialogs = dialogs
        self.contexts = {"context": []}
        self.logger.info("building data from segments")
        data = [self.build_data(dialog=dialog, vocab=vocab, max_history_utterance=max_history_utterance,
                                max_seq_len=max_seq_len)
                for dialog in dialogs]
        self.logger.info("padding and converting to tensor")
        self.pad_data = self.get_padding_data(data)

    def __len__(self):
        return self.pad_data["context"].shape[0]

    def __getitem__(self, item):
        return {"context": self.pad_data["context"][item, :],  # context_len
                "context_type_ids": self.pad_data["context_type_ids"][item, :],  # context_len
                "context_position_ids": self.pad_data["context_position_ids"][item, :],  # context_len
                "context_utterance_position_ids": self.pad_data["context_utterance_position_ids"][item, :],  # context_len
                "tgt": self.pad_data["tgt"][item, :],  # tgt_len
                "tgt_type_ids": self.pad_data["tgt_type_ids"][item, :],  # tgt_len
                "tgt_position_ids": self.pad_data["tgt_position_ids"][item, :],  # tgt_len
                "tgt_utterance_position_ids": self.pad_data["tgt_utterance_position_ids"][item, :],  # tgt_len
                }

    def build_data(self, dialog, vocab, max_history_utterance, max_seq_len):
        self.contexts["context"].append(dialog['context'])
        context = dialog['context'][-max_history_utterance:]
        context = [seq[:max_seq_len - 1] + [vocab.eos_id] for seq in context]
        context_utterance_position_ids = [len(context) - utter_idx
                                          for utter_idx, seq in enumerate(context) for _ in seq]
        context_type_ids = [vocab.human_id if (len(context) - i) % 2 else vocab.bot_id
                            for i, s in enumerate(context) for _ in s]
        context_position_ids = [i for s in context for i in range(len(s))]
        context = list(chain(*context))
        tgt = [vocab.bos_id] + dialog['tgt'][: max_seq_len - 2] + [vocab.eos_id]
        tgt_type_ids = [vocab.bot_id] * len(tgt)
        tgt_position_ids = list(range(len(tgt)))
        tgt_utterance_position_ids = [0 for _ in tgt_position_ids]

        return {"context": context,
                "context_type_ids": context_type_ids,
                "context_position_ids": context_position_ids,
                "context_utterance_position_ids": context_utterance_position_ids,
                "tgt": tgt,
                "tgt_type_ids": tgt_type_ids,
                "tgt_position_ids": tgt_position_ids,
                "tgt_utterance_position_ids": tgt_utterance_position_ids}

    def get_padding_data(self, data):
        pad_data = {"context": [],  # n_samples, context_len
                    "context_type_ids": [],  # n_samples, context_len
                    "context_position_ids": [],  # n_samples, context_len
                    "context_utterance_position_ids": [],  # n_samples, context_len
                    "tgt": [],  # n_samples, tgt_len
                    "tgt_type_ids": [],  # n_samples, tgt_len
                    "tgt_position_ids": [],  # n_samples, tgt_len
                    "tgt_utterance_position_ids": [],  # n_samples, tgt_len
                    }
        for instance in data:
            for key_name in instance:
                pad_data[key_name].append(instance[key_name])

        # pad context
        max_context_len = max(len(sequence) for sequence in pad_data["context"])
        for key in ["context", "context_type_ids"]:
            pad_data[key] = self.pad_and_convert_to_tensor(pad_data[key], pad_id=self.vocab.pad_id,
                                                           max_seq_len=max_context_len)
        for key in ["context_position_ids", "context_utterance_position_ids"]:
            pad_data[key] = self.pad_and_convert_to_tensor(pad_data[key], pad_id=0, max_seq_len=max_context_len)

        # pad tgt
        max_tgt_len = max(len(sequence) for sequence in pad_data["tgt"])
        for key in ["tgt", "tgt_type_ids"]:
            pad_data[key] = self.pad_and_convert_to_tensor(pad_data[key], pad_id=self.vocab.pad_id,
                                                           max_seq_len=max_tgt_len)
        for key in ["tgt_position_ids", "tgt_utterance_position_ids"]:
            pad_data[key] = self.pad_and_convert_to_tensor(pad_data[key], pad_id=0, max_seq_len=max_tgt_len)

        return {
            "context": pad_data["context"],
            "context_type_ids": pad_data["context_type_ids"],
            "context_position_ids": pad_data["context_position_ids"],
            "context_utterance_position_ids": pad_data["context_utterance_position_ids"],
            "tgt": pad_data["tgt"],
            "tgt_type_ids": pad_data["tgt_type_ids"],
            "tgt_position_ids": pad_data["tgt_position_ids"],
            "tgt_utterance_position_ids": pad_data["tgt_utterance_position_ids"]
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


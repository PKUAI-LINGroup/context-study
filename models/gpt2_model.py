# -*- coding: utf-8 -*-
# @File        : gpt2_model.py
# @Description :


import torch.nn as nn


class GPT2Model(nn.Module):
    def __init__(self, core_module, vocab):
        super(GPT2Model, self).__init__()
        self.core_module = core_module
        self.vocab = vocab
        self.pad_id = vocab.pad_id
        # init lm head
        embed_dim = core_module.wte.weight.size(1)
        vocab_size = core_module.wte.weight.size(0)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.lm_head.weight = core_module.wte.weight
        # init loss function
        self.lm_criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)

    def forward(self, input_ids, type_ids, position_ids, utterance_position_ids, lm_labels=None,
                past=None, return_past=False):
        hidden_states, past = self.core_module(input_ids=input_ids, token_type_ids=type_ids, past=past,
                                               position_ids=position_ids,
                                               utterance_position_ids=utterance_position_ids)
        lm_logits = self.lm_head(hidden_states)
        output = {"lm_logits": lm_logits}
        if return_past:
            output["past"] = past
        if lm_labels is not None:
            lm_loss = self.lm_criterion(lm_logits[:, :-1, :].contiguous().view(-1, lm_logits.shape[-1]),
                                        lm_labels[:, 1:].contiguous().view(-1))
            output["lm_loss"] = lm_loss
        return output

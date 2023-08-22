# -*- coding: utf-8 -*-
# @Time        : 2023/8/23 01:23
# @Author      : ssxy00
# @File        : trm_model.py
# @Description :


import torch.nn as nn
from modules.trm_encoder_module import TrmEncoderModule
from modules.trm_decoder_module import TrmDecoderModule


class TrmEncoder(nn.Module):
    def __init__(self, core_module, vocab):
        super(TrmEncoder, self).__init__()
        self.core_module = core_module
        self.vocab = vocab
        self.pad_id = vocab.pad_id

    def forward(self, context, context_type_ids, context_position_ids, context_utterance_position_ids):
        """
        utterance encoder
        """
        attention_mask = context.ne(self.pad_id).float()
        encoder_out = self.core_module(input_ids=context, token_type_ids=context_type_ids,
                                       position_ids=context_position_ids,
                                       utterance_position_ids=context_utterance_position_ids,
                                       attention_mask=attention_mask)
        return encoder_out[0], attention_mask


class TrmDecoder(nn.Module):
    def __init__(self, core_module, vocab):
        super(TrmDecoder, self).__init__()
        self.core_module = core_module
        self.vocab = vocab
        self.pad_id = vocab.pad_id
        # init lm head
        embed_dim = core_module.wte.weight.size(1)
        vocab_size = core_module.wte.weight.size(0)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.lm_head.weight = core_module.wte.weight

    def forward(self, tgt, tgt_type_ids, tgt_position_ids, tgt_utterance_position_ids,
                encoder_memory, cross_attention_mask, past=None, return_past=False):
        hidden_states, past = self.core_module(input_ids=tgt, token_type_ids=tgt_type_ids, past=past,
                                               position_ids=tgt_position_ids,
                                               utterance_position_ids=tgt_utterance_position_ids,
                                               encoder_memory=encoder_memory,
                                               cross_attention_mask=cross_attention_mask)
        lm_logits = self.lm_head(hidden_states)
        if return_past:
            return lm_logits, past
        return lm_logits


class TrmModel(nn.Module):
    def __init__(self, model_config, vocab, args):
        super(TrmModel, self).__init__()
        encoder_module = TrmEncoderModule(model_config.encoder_config,
                                          n_utterance_positions=args.max_history_utterance + 1)
        decoder_module = TrmDecoderModule(model_config.decoder_config,
                                          n_utterance_positions=args.max_history_utterance + 1)
        decoder_module.wte.weight = encoder_module.wte.weight
        decoder_module.wpe.weight = encoder_module.wpe.weight
        decoder_module.upe.weight = encoder_module.upe.weight

        self.encoder = TrmEncoder(core_module=encoder_module, vocab=vocab)
        self.decoder = TrmDecoder(core_module=decoder_module, vocab=vocab)

        # init loss function
        self.lm_criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)

    def forward(self, context, context_type_ids, context_position_ids, context_utterance_position_ids,
                tgt, tgt_type_ids, tgt_position_ids, tgt_utterance_position_ids, lm_labels=None):
        encoder_memory, cross_attention_mask = self.encoder(
            context=context, context_type_ids=context_type_ids, context_position_ids=context_position_ids,
            context_utterance_position_ids=context_utterance_position_ids)

        lm_logits = self.decoder(tgt=tgt, tgt_type_ids=tgt_type_ids, tgt_position_ids=tgt_position_ids,
                                 tgt_utterance_position_ids=tgt_utterance_position_ids,
                                 encoder_memory=encoder_memory, cross_attention_mask=cross_attention_mask)

        if lm_labels is not None:
            lm_loss = self.lm_criterion(lm_logits[:, :-1, :].contiguous().view(-1, lm_logits.shape[-1]),
                                        lm_labels[:, 1:].contiguous().view(-1))
            return {"lm_logits": lm_logits, "lm_loss": lm_loss}
        return {"lm_logits": lm_logits}


# -*- coding: utf-8 -*-
# @File        : evaluate_utils.py
# @Description :


import torch


def greedy_search_for_trm(context, context_type_ids, context_position_ids, context_utterance_position_ids,
                          max_len, vocab, model):
    with torch.no_grad():
        batch_size = context.shape[0]
        device = context.device

        is_end = torch.zeros(batch_size, dtype=torch.bool, device=device)

        results = torch.full((batch_size, 1), fill_value=vocab.bos_id, dtype=torch.long, device=device)
        prevs = torch.full((batch_size, 1), fill_value=vocab.bos_id, dtype=torch.long, device=device)
        past = None

        tgt_type_ids = torch.full(prevs.shape, fill_value=vocab.bot_id, dtype=torch.long, device=device)
        tgt_position_ids = torch.full(prevs.shape, fill_value=0, dtype=torch.long, device=device)
        tgt_utterance_position_ids = torch.full(prevs.shape, fill_value=0, dtype=torch.long, device=device)

        encoder_memory, cross_attention_mask = model.encoder(
            context=context, context_type_ids=context_type_ids, context_position_ids=context_position_ids,
            context_utterance_position_ids=context_utterance_position_ids)

        for i in range(max_len):
            logits, past = model.decoder(tgt=prevs, tgt_type_ids=tgt_type_ids, tgt_position_ids=tgt_position_ids,
                                         tgt_utterance_position_ids=tgt_utterance_position_ids,
                                         encoder_memory=encoder_memory, cross_attention_mask=cross_attention_mask,
                                         past=past, return_past=True)
            selected_idxs = logits[:, -1, :].argmax(-1)
            selected_idxs[is_end] = vocab.pad_id
            is_end[selected_idxs == vocab.eos_id] = 1  # <eos> means end of sentence
            prevs = selected_idxs.unsqueeze(-1)
            tgt_position_ids += 1
            results = torch.cat([results, selected_idxs.unsqueeze(-1)], dim=1)

            if all(is_end.view(-1)):
                break
    return results


def greedy_search_for_gpt2(input_ids, type_ids, position_ids, utterance_position_ids, max_len, vocab, model):
    with torch.no_grad():
        batch_size = input_ids.shape[0]
        device = input_ids.device

        is_end = torch.zeros(batch_size, dtype=torch.bool, device=device)

        results = torch.full((batch_size, 1), fill_value=vocab.bos_id, dtype=torch.long, device=device)
        prevs = torch.full((batch_size, 1), fill_value=vocab.bos_id, dtype=torch.long, device=device)
        past = None

        tgt_type_ids = torch.full(prevs.shape, fill_value=vocab.bot_id, dtype=torch.long, device=device)
        tgt_position_ids = torch.full(prevs.shape, fill_value=0, dtype=torch.long, device=device)
        tgt_utterance_position_ids = torch.full(prevs.shape, fill_value=0, dtype=torch.long, device=device)

        _, past = model.core_module(input_ids=input_ids, token_type_ids=type_ids, past=past,
                                    position_ids=position_ids,
                                    utterance_position_ids=utterance_position_ids)

        for i in range(max_len):
            output = model(input_ids=prevs, type_ids=tgt_type_ids, position_ids=tgt_position_ids,
                           utterance_position_ids=tgt_utterance_position_ids,
                           past=past, return_past=True)
            past = output["past"]
            selected_idxs = output["lm_logits"][:, -1, :].argmax(-1)
            selected_idxs[is_end] = vocab.pad_id
            is_end[selected_idxs == vocab.eos_id] = 1  # <eos> means end of sentence
            prevs = selected_idxs.unsqueeze(-1)
            tgt_position_ids += 1
            results = torch.cat([results, selected_idxs.unsqueeze(-1)], dim=1)

            if all(is_end.view(-1)):
                break
    return results

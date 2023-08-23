max_history_utterance=13
lr=8e-5
ckpt=3

checkpoint_path=./checkpoints/gpt2/personachat/${max_history_utterance}/${lr}/checkpoint${ckpt}.pt
save_result_path=./results/gpt2_personachat_${max_history_utterance}_${lr}_${ckpt}.jsonl
CUDA_VISIBLE_DEVICES=0 python evaluate_gpt2.py \
--test_dataset ./processed_data/personachat/test.cache \
--max_history_utterance $max_history_utterance \
--max_seq_len 50 \
--max_predict_len 50 \
--checkpoint_path $checkpoint_path \
--save_result_path $save_result_path

max_history_utterance=10
max_seq_len=50
lr=5e-4

TRAINDATA=./processed_data/dailydialog/train.cache
VALIDDATA=./processed_data/dailydialog/valid.cache

MODEL_DIR=./checkpoints/trm/dailydialog/$lr
LOG_DIR=./logs/trm/dailydialog/$lr
mkdir -p $MODEL_DIR
mkdir -p $LOG_DIR

log_path=train_logs/trm_dailydialog_${max_history_utterance}_${lr}.log
echo $MODEL_DIR >> $log_path

CUDA_VISIBLE_DEVICES=0 python train_trm.py \
--train_dataset $TRAINDATA --valid_dataset $VALIDDATA \
--save_model_dir $MODEL_DIR \
--save_interval 1 \
--log_dir $LOG_DIR \
--max_seq_len $max_seq_len \
--max_history_utterance $max_history_utterance \
--n_epochs 10 \
--batch_size 64 --gradient_accumulate_steps 1 \
--lr $lr >> $log_path
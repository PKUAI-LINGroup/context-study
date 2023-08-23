# context-study
The official repo of the PRICAI-2023 short paper titled "An Empirical Study on Context Length for Open-Domain Dialog Generation".

## Environment
```
conda create -n context python==3.6.8
conda activate context
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
python -m pip install transformers==3.0.2
python -m pip install tensorboard
python -m pip install jsonlines
```

## Data preparation
### Tokenizer
We use GPT2 vocabulary in our experiments. To prepare vocabulary files, please:
+ download `gpt2-vocab.json` from [here](https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json), rename it to `vocab.json`, and move it to the folder `./gpt2_vocab/`
+ download `gpt2-merges.txt` from [here](https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt), rename it to `merges.txt`, and move it to the folder `./gpt2_vocab/`

### Datasets
Experiments are conducted on two widely used open-domain dialog datasets:
+ [PersonaChat](https://aclanthology.org/P18-1205/)
+ [DalyDialog](https://aclanthology.org/I17-1099/)

After downloading raw data, please run scripts in `./prepare_data/` to preprocess data. The statistics of the processed data are shown in the table below.

| dataset     | train   | valid | test |
|-------------|---------|-------|------|
| DailyDialog | 76052   | 7069  | 6740 |
| PersonaChat | 244998  | 7290  | 7312 |


## Models
We build two dialog models: one trained from scratch on Transformer and the other fine-tuned on GPT2.

+ Transformer: 3 encoder layers, 3 decoder layers, 2 heads, 256 dimensions
+ GPT2: 12 layers, 12 heads, 768 dimensions

We ues the pre-trained parameters released by [HuggingFace](https://github.com/huggingface/transformers). To prepare GPT2 model files, please download `config.json` and `pytorch_model.bin` from [here](https://huggingface.co/gpt2/tree/main) and move them to the folder `./gpt2_model/`.

The input representation is shown in the figure below.
![image](https://github.com/PKUAI-LINGroup/context-study/blob/main/figs/dialog_model_input_representation.png)

## Training scripts
+ Training Transformer on PersonaChat: `bash scripts/train_trm_personachat.sh`
+ Training Transformer on DailyDialog: `bash scripts/train_trm_dailydialog.sh`
+ Fine-tuning GPT2 on PersonaChat: `bash scripts/train_gpt2_personachat.sh`
+ Fine-tuning GPT2 on DailyDialog: `bash scripts/train_gpt2_dailydialog.sh`

## Evaluation scripts
+ Evaluating Transformer on PersonaChat: `bash scripts/evaluate_trm_personachat.sh`
+ Evaluating Transformer on DailyDialog: `bash scripts/evaluate_trm_dailydialog.sh`
+ Evaluating GPT2 on PersonaChat: `bash scripts/evaluate_gpt2_personachat.sh`
+ Evaluating GPT2 on DailyDialog: `bash scripts/evaluate_gpt2_dailydialog.sh`
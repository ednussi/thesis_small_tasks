# Set env
virtualenv text-mosaic-augs -p python3
pip install git+https://github.com/huggingface/transformers
pip install nltk datasets
pip install rouge_score

## run locally
### Sumri
python run_summarization.py --model_name_or_path t5-small --do_train --do_eval --dataset_name xsum --source_prefix "summarize: " --output_dir "/cs/labs/gabis/ednussi/thesis_small_tasks/sumri_res_test" --per_device_train_batch_size=4 --per_device_eval_batch_size=4 --overwrite_output_dir --predict_with_generate --warmup_ratio=0.1 --max_train_samples 1000000 --num_train_epochs=10 --seed 42 --aug crop

### NER
python run_ner.py --model_name_or_path bert-base-uncased --dataset_name wnut_17 --do_train --do_eval --warmup_ratio=0.1 --overwrite_output_dir --num_train_epochs=10 --per_gpu_train_batch_size=8 --seed 42 --max_train_samples 64 --output_dir test_res --aug concat

### QA
python run_qa.py --model_name_or_path roberta-base --do_train --do_eval --dataset_name $d --output_dir $OUTPUTDIR --per_device_train_batch_size=4 --per_device_eval_batch_size=4 --overwrite_output_dir --warmup_ratio=0.1 --max_train_samples $i --num_train_epochs=10 --seed $j --aug $aug --save_steps=50000

<!-- ==== THIS IS A COMMENT ====
# SWAG
python run_swag.py \
    --model_name_or_path roberta-base \
    --do_train \
    --do_eval \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --output_dir /tmp/swag_base \
    --per_gpu_eval_batch_size=16 \
    --per_device_train_batch_size=16 \
    --overwrite_output \
    --warmup_ratio=0.1 \
    --max_train_samples 16 \
    --num_train_epochs=10 \
    --seed 42 \
    --aug baseline


# Coreference
## Install pacakges
Install neuralcoref
```
git clone https://github.com/huggingface/neuralcoref.git
cd neuralcoref
pip install -r requirements.txt
pip install -e .
```
```
pip install -U spacy
python -m spacy download en
```
==== THIS IS A COMMENT ==== -->

# Datasets
## NER

| Dataset      | Full Dataset |
|--------------|--------------|
| wnut_17      | 3394         |
| conll2003    | 14041        |
| ncbi_disease | 5433         |
| species_800  | 5734         |
|bc2gm_corpus  |   12501      |

## Abstractive Summrization

| Dataset      | Full Dataset |
|--------------|--------------|
| xsum      | 204045         |
| samsum    | 14732        |

## Extractive Question Answering

| Dataset      | Full Dataset |
|--------------|--------------|
| squad      | 87599         |
| hotpot_qa    | 90447        |

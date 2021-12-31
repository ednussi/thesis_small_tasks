# Set env
virtualenv sumri -p python3
pip install git+https://github.com/huggingface/transformers
pip install nltk datasets
pip install rouge_score
# SUMMARIZATION
python run_summarization.py \
    --model_name_or_path t5-small \
    --do_train \
    --do_eval \
    --dataset_name xsum \
    --source_prefix "summarize: " \
    --output_dir /tmp/tst-summarization \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate \
    --warmup_ratio=0.1 \
    --max_train_samples 16 \
    --num_train_epochs=10 \
    --seed 42 \
    --aug baseline


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

# NER
python run_ner.py --model_name_or_path bert-base-uncased --dataset_name wnut_17 --do_train --do_eval --warmup_ratio=0.1 --overwrite_output_dir --num_train_epochs=10 --seed 42 --max_train_samples 16 --output_dir test_res --aug mosaic-crop

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


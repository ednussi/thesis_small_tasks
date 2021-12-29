# SUMMARIZATION
Requires installation of huggingface from source pip install git+https://github.com/huggingface/transformers

python run_summarization.py \
    --model_name_or_path t5-small \
    --do_train \
    --do_eval \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir /tmp/tst-summarization \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate

Extreme Summarization (XSum) Dataset is another commonly used dataset for the task of summarization. To use it replace --dataset_name cnn_dailymail --dataset_config "3.0.0" with --dataset_name xsum.


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


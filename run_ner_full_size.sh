#!/bin/sh
export MODEL="bert-base-uncased"

for j in 42 43 44 45 46
do
  for i in 15000
  do
    for d in 'conll2003' 'wnut_17' 'ncbi_disease' 'species_800' 'bc2gm_corpus'
    do
      for aug in 'baseline' 'double-baseline' 'crop' 'mosaic' 'mosaic-crop'
      do
        echo "===============Dataset $d Loop $i-$j Aug $aug==============="
        EXPNAME="$d-$aug"
        OUTPUTDIR="/d/Thesis/thesis_small_tasks/ner_res/$EXPNAME/output-$i-$j"
        mkdir -p -- $OUTPUTDIR
        python run_ner.py --model_name_or_path $MODEL --dataset_name $d --do_train --do_eval --warmup_ratio=0.1 --overwrite_output_dir --num_train_epochs=10 --seed $j --max_train_samples $i --output_dir $OUTPUTDIR --aug $aug
      done
    done
  done
done
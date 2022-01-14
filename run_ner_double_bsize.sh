#!/bin/sh
export MODEL="bert-base-uncased"
for d in 'conll2003' 'wnut_17' 'ncbi_disease' 'species_800' 'bc2gm_corpus'
do
  for i in 256 128 64 32 16
  do
    for j in 42 43 44 45 46
    do
      for aug in 'double-baseline'
      do
        echo "===============Dataset $d Loop $i-$j Aug $aug==============="
        EXPNAME="$d-$aug"
        OUTPUTDIR="/d/Thesis/thesis_small_tasks/ner_res/$EXPNAME/output-$i-$j"
        mkdir -p -- $OUTPUTDIR
        python run_ner.py --model_name_or_path $MODEL --dataset_name $d --do_train --do_eval --warmup_ratio=0.1 --per_gpu_train_batch_size=16 --overwrite_output_dir --num_train_epochs=10 --seed $j --max_train_samples $i --output_dir $OUTPUTDIR --aug $aug
        rm -rf "$OUTPUTDIR/pytorch_model.bin"
        rm -rf "$OUTPUTDIR/optimizer.pt"
        rm -rf "$OUTPUTDIR/output-$i-$j/checkpoint*/optimizer.pt"
        rm -rf "$OUTPUTDIR/output-$i-$j/checkpoint*/pytorch_model.bin"
      done
    done
  done
done
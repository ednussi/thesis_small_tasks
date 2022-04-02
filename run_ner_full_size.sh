#!/bin/sh
export MODEL="bert-base-uncased"

for j in 42 43 44 45 46
do
  for i in 15000
  do
    for aug in 'crop' 'mosaic' 'double-baseline'
    do
      for d in 'conll2003' 'wnut_17' 'ncbi_disease' 'species_800' 'bc2gm_corpus'
      do
        echo "===============Dataset $d Loop $i-$j Aug $aug==============="
        EXPNAME="$d-$aug"
        OUTPUTDIR="/d/Thesis/thesis_small_tasks/ner_full_res_new/$EXPNAME/output-$i-$j"
        mkdir -p -- $OUTPUTDIR
        python run_ner.py --model_name_or_path $MODEL --dataset_name $d --do_train --do_eval --warmup_ratio=0.1 --overwrite_output_dir --num_train_epochs=10 --seed $j --max_train_samples $i --output_dir $OUTPUTDIR --aug $aug --save_steps=10000
        rm -rf "$OUTPUTDIR/pytorch_model.bin"
        rm -rf "$OUTPUTDIR/optimizer.pt"
        rm -rf "$OUTPUTDIR/output-$i-$j/pytorch_model.bin"
        rm -rf "$OUTPUTDIR/output-$i-$j/optimizer.pt"
        rm -rf "$OUTPUTDIR/checkpoint*/optimizer.pt"
        rm -rf "$OUTPUTDIR/checkpoint*/pytorch_model.bin"
      done
    done
  done
done

for j in 42 43 44 45 46
do
  for i in 8192
  do
    for aug in 'baseline' 'mosaic-crop' 'crop' 'mosaic' 'double-baseline'
    do
      for d in 'conll2003' 'bc2gm_corpus'
      do
        echo "===============Dataset $d Loop $i-$j Aug $aug==============="
        EXPNAME="$d-$aug"
        OUTPUTDIR="/d/Thesis/thesis_small_tasks/ner_full_res_new/$EXPNAME/output-$i-$j"
        mkdir -p -- $OUTPUTDIR
        python run_ner.py --model_name_or_path $MODEL --dataset_name $d --do_train --do_eval --warmup_ratio=0.1 --overwrite_output_dir --num_train_epochs=10 --seed $j --max_train_samples $i --output_dir $OUTPUTDIR --aug $aug --save_steps=5000
        rm -rf "$OUTPUTDIR/pytorch_model.bin"
        rm -rf "$OUTPUTDIR/optimizer.pt"
        rm -rf "$OUTPUTDIR/output-$i-$j/pytorch_model.bin"
        rm -rf "$OUTPUTDIR/output-$i-$j/optimizer.pt"
        rm -rf "$OUTPUTDIR/checkpoint*/optimizer.pt"
        rm -rf "$OUTPUTDIR/checkpoint*/pytorch_model.bin"
      done
    done
  done
done

for j in 42 43 44 45 46
do
  for i in 4096
  do
    for aug in 'baseline' 'mosaic-crop' 'crop' 'mosaic' 'double-baseline'
    do
      for d in 'conll2003' 'wnut_17' 'ncbi_disease' 'species_800' 'bc2gm_corpus'
      do
        echo "===============Dataset $d Loop $i-$j Aug $aug==============="
        EXPNAME="$d-$aug"
        OUTPUTDIR="/d/Thesis/thesis_small_tasks/ner_full_res_new/$EXPNAME/output-$i-$j"
        mkdir -p -- $OUTPUTDIR
        python run_ner.py --model_name_or_path $MODEL --dataset_name $d --do_train --do_eval --warmup_ratio=0.1 --overwrite_output_dir --num_train_epochs=10 --seed $j --max_train_samples $i --output_dir $OUTPUTDIR --aug $aug --save_steps=3000
        rm -rf "$OUTPUTDIR/pytorch_model.bin"
        rm -rf "$OUTPUTDIR/optimizer.pt"
        rm -rf "$OUTPUTDIR/output-$i-$j/pytorch_model.bin"
        rm -rf "$OUTPUTDIR/output-$i-$j/optimizer.pt"
        rm -rf "$OUTPUTDIR/checkpoint*/optimizer.pt"
        rm -rf "$OUTPUTDIR/checkpoint*/pytorch_model.bin"
      done
    done
  done
done

for j in 42 43 44 45 46
do
  for i in 2048 1024 512 256 128 64 32 16 8 4 2 1
  do
    for aug in 'baseline' 'mosaic-crop' 'crop' 'mosaic' 'double-baseline'
    do
      for d in 'conll2003' 'ncbi_disease' 'species_800' 'bc2gm_corpus'
      do
        echo "===============Dataset $d Loop $i-$j Aug $aug==============="
        EXPNAME="$d-$aug"
        OUTPUTDIR="/d/Thesis/thesis_small_tasks/ner_full_res_new/$EXPNAME/output-$i-$j"
        mkdir -p -- $OUTPUTDIR
        python run_ner.py --model_name_or_path $MODEL --dataset_name $d --do_train --do_eval --warmup_ratio=0.1 --overwrite_output_dir --num_train_epochs=10 --seed $j --max_train_samples $i --output_dir $OUTPUTDIR --aug $aug --save_steps=1200
        rm -rf "$OUTPUTDIR/pytorch_model.bin"
        rm -rf "$OUTPUTDIR/optimizer.pt"
        rm -rf "$OUTPUTDIR/output-$i-$j/pytorch_model.bin"
        rm -rf "$OUTPUTDIR/output-$i-$j/optimizer.pt"
        rm -rf "$OUTPUTDIR/checkpoint*/optimizer.pt"
        rm -rf "$OUTPUTDIR/checkpoint*/pytorch_model.bin"
      done
    done
  done
done
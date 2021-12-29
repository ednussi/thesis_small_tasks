#!/bin/sh
#SBATCH --time=2-0:0:0
#SBATCH --gres=gpu:rtx2080:1
source  /cs/labs/gabis/ednussi/ner/bin/activate
export MODEL="roberta-base"
export BASEPATH="/cs/labs/gabis/ednussi/splinter/finetuning"
for d in 'conll2003' 'wnut_17' 'ncbi_disease' 'species_800' 'bc2gm_corpus'
do
  for i in 2048 1024 512 256 128 64 32 16
  do
    for j in 42 43 44 45 46
    do
      echo "Loop $i-$j"
      AUG="mosaic"
      EXPNAME="$d-$AUG"
      OUTPUTDIR="$BASEPATH/results_ner/$EXPNAME/output-$i-$j"
      mkdir -p -- $OUTPUTDIR
      python run_ner.py --model_name_or_path bert-base-uncased --dataset_name $d --do_train --do_eval --warmup_ratio=0.1 --overwrite_output_dir --num_train_epochs=10 --seed $j --max_train_samples $i --output_dir $OUTPUTDIR --aug $AUG
      rm -rf "$OUTPUTDIR/pytorch_model.bin"
    done
  done
done
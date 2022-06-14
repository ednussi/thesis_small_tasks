#!/bin/sh
for d in 'squad'
do
  for j in 42 43 44 45 46
  do
    for i in 16 32 64 128 256
    do
      for aug in 'concat' 'crop' 'mosaic'
      do
        echo "=============== Dataset $d, Aug $aug, Examples $i, Seed $j ==============="
        EXPNAME="$d-$aug"
        EXPPATH="/d/Thesis/thesis_small_tasks/qa_res_all_augs/$EXPNAME"
        OUTPUTDIR="$EXPPATH/output-$i-$j"
        mkdir -p -- $OUTPUTDIR
        python run_qa.py --model_name_or_path roberta-base --do_train --do_eval --dataset_name $d --output_dir $OUTPUTDIR --per_device_train_batch_size=4 --per_device_eval_batch_size=4 --overwrite_output_dir --warmup_ratio=0.1 --max_train_samples $i --num_train_epochs=10 --seed $j --aug $aug --save_steps=50000
        rm -rf "$EXPPATH/pytorch_model.bin"
        rm -rf "$EXPPATH/optimizer.pt"
        rm -rf "$OUTPUTDIR/checkpoint*/optimizer.pt"
        rm -rf "$OUTPUTDIR/checkpoint*/pytorch_model.bin"
      done
    done
  done
done
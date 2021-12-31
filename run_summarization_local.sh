#!/bin/sh
for d in 'xsum' 'samsum'
do
  for aug in 'baseline' 'mosaic' 'lorem-ipsum' 'lorem-ipsum-double'
  do
    for i in 1024 512 256 128 64 32 16
    do
      for j in 42 43 44 45 46
      do
        echo "=============== Dataset $d, Aug $aug, Examples $i, Seed $j ==============="
        EXPNAME="$d-$aug"
        OUTPUTDIR="/d/Thesis/thesis_small_tasks/sumri_res/$EXPNAME/output-$i-$j"
        mkdir -p -- $OUTPUTDIR
        python run_summarization.py --model_name_or_path t5-small --do_train --do_eval --dataset_name $d --source_prefix "summarize: " --output_dir $OUTPUTDIR --per_device_train_batch_size=4 --per_device_eval_batch_size=4 --overwrite_output_dir --predict_with_generate --warmup_ratio=0.1 --max_train_samples $i --num_train_epochs=10 --seed $j --aug $aug
        rm -rf "$OUTPUTDIR/pytorch_model.bin"
        rm -rf "$OUTPUTDIR/output-$i-$j/checkpoint*/optimizer.pt"
        rm -rf "$OUTPUTDIR/output-$i-$j/checkpoint*/pytorch_model.bin"
      done
    done
  done
done
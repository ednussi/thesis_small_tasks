#!/bin/sh
#SBATCH --time=7-0:0:0
#SBATCH --gres=gpu:rtx6000:1
source  /cs/labs/gabis/ednussi/sumri/bin/activate

for d in 'xsum' 'samsum'
do
  for j in 44
  do
    for i in 1000000
    do
      for aug in 'mosaic' 'crop' 'concat' 'double-baseline' 'baseline'
      do
        echo "=============== Dataset $d, Aug $aug, Examples $i, Seed $j ==============="
        EXPNAME="$d-$aug"
        EXPPATH="/cs/labs/gabis/ednussi/thesis_small_tasks/swag_res/$EXPNAME"
        OUTPUTDIR="$EXPPATH/output-$i-$j"
        mkdir -p -- $OUTPUTDIR
        python run_summarization.py --model_name_or_path t5-small --do_train --do_eval --dataset_name $d --source_prefix "summarize: " --output_dir $OUTPUTDIR --per_device_train_batch_size=4 --per_device_eval_batch_size=4 --overwrite_output_dir --predict_with_generate --warmup_ratio=0.1 --max_train_samples $i --num_train_epochs=10 --seed $j --aug $aug --save_steps=250000
        rm -rf "$EXPPATH/pytorch_model.bin"
        rm -rf "$EXPPATH/optimizer.pt"
        rm -rf "$OUTPUTDIR/checkpoint*/optimizer.pt"
        rm -rf "$OUTPUTDIR/checkpoint*/pytorch_model.bin"
      done
    done
  done
done
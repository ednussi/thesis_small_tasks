#!/bin/sh
#SBATCH --time=2-0:0:0
#SBATCH --gres=gpu:rtx2080:1
source  /cs/labs/gabis/ednussi/sumri/bin/activate

for aug in 'lorem-ipsum-context' 'lorem-ipsum-answers'
do
  for i in 1024 512
  do
    for j in 42 43 44 45 46
    do
      echo "=============== Dataset SWAG, Aug $aug, Examples $i, Seed $j ==============="
      EXPNAME="swag-$aug"
      OUTPUTDIR="/cs/labs/gabis/ednussi/thesis_small_tasks/swag_res/$EXPNAME/output-$i-$j"
      mkdir -p -- $OUTPUTDIR
      python run_swag.py --model_name_or_path roberta-base --do_train --do_eval --learning_rate 5e-5 --num_train_epochs 10 --output_dir $OUTPUTDIR --per_gpu_eval_batch_size=16 --per_device_train_batch_size=16 --overwrite_output --warmup_ratio=0.1 --max_train_samples $i --seed $j --aug $aug
      rm -rf "$OUTPUTDIR/pytorch_model.bin"
      rm -rf "$OUTPUTDIR/output-$i-$j/checkpoint*/optimizer.pt"
      rm -rf "$OUTPUTDIR/output-$i-$j/checkpoint*/pytorch_model.bin"
    done
  done
done
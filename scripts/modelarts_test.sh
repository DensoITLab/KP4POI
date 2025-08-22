#!/bin/bash

task=$1
test_prompt=$2
outdir=$3

#python -m torch.distributed.launch --nproc_per_node=4 --master_port 12345 
python src/test_sequential.py \
    --batch_size 8 \
    --hop_num 2 \
    --degree 8 \
    --max_items 5 \
    --test ${task} \
    --tree_mask \
    --data_url data \
    --train_url ${outdir} \
    --load ${outdir}/BEST_EVAL_LOSS.pth \
    --num_workers 2 \
    --test_prompt ${test_prompt}

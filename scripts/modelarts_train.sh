#!/bin/bash

task=$1

num_workers=0
batch_size=64
epoch=100
if [ 'flm-1b' == $task ]; then
    num_workers=0
fi
if [ 'amazon-books' == $task ]; then
    epoch=60
fi

#python -m torch.distributed.launch --nproc_per_node=1 \
python  src/main.py \
    --prefix _2hop_d8_L5_1e-3_mask \
    --batch_size ${batch_size} \
    --hop_num 2 \
    --degree 8 \
    --max_items 5 \
    --lr 1e-3 \
    --epoch ${epoch} \
    --train ${task} \
    --valid ${task} \
    --tree_mask \
    --data_url data \
    --log_url log \
    --num_workers ${num_workers}

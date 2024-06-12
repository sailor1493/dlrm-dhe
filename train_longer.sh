#!/bin/bash

emb=128
# repeat 26 times, 128,128,128, ...
# omit the last comma
embs=$(printf "%s," $(for i in $(seq 1 26); do echo $emb; done) | sed 's/,$//')

DISTRIBUTED_ARGS="
    --nproc_per_node 2 \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr localhost \
    --master_port 5678
"
power=5
bs=$((2**$power))

tbs=$((2**18))

# log filename: train-date.log
filename="train-$(date +'%Y-%m-%d-%H-%M-%S').log"
echo "Logging to $filename"

torchrun $DISTRIBUTED_ARGS dlrm_main.py \
    --epochs 1 \
    --batch_size $bs \
    --drop_last_training_batch \
    --test_batch_size $tbs \
    --limit_train_batches 16384 \
    --dataset_name criteo_kaggle \
    --num_embeddings 100_000 \
    --num_embeddings_per_feature $embs \
    --dense_arch_layer_sizes 512,256,64 \
    --over_arch_layer_sizes 512,512,256,1 \
    --embedding_dim 64 \
    --interaction_branch1_layer_sizes 2048,2048 \
    --interaction_branch2_layer_sizes 2048,2048 \
    --dcn_num_layers 3 \
    --dcn_low_rank_dim 512 \
    --seed 42 \
    --pin_memory \
    --mmap_mode \
    --learning_rate 1 \
    --in_memory_binary_criteo_path /root/dlrm-dhe/torchrec_dlrm/SSD/display-preprocessed \
    --lr_warmup_steps 4096 \
    --lr_decay_start 4096 \
    --lr_decay_steps 12288 \
    --validation_freq_within_epoch 2048 | tee $filename
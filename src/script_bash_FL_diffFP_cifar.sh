#!/bin/bash
# Comments line start with a #
# Commands are surrounde by ()
# Website on how to write bash script https://hackernoon.com/know-shell-scripting-202b2fbe03a8

# Set GPU device (you can ignore this if not usin GPU)
GPU_ID="cuda:0"



# ================ 32-bit ================ 
# This is for FL for 32-bit floating point
python federated_main.py --local_ep=5 --local_bs=50 --frac=0.1 --model=cnn --dataset=cifar --iid=1 --gpu=1 --lr=0.01 --epochs=300


# This is for 2 clusters HFL for 32-bit floating point
python federated-hierarchical2_main.py --local_ep=5 --local_bs=50 --frac=0.1 --Cepochs=10 --model=cnn --dataset=cifar --iid=1 --num_cluster=2 --gpu=1 --lr=0.01 --epochs=100


# This is for 4 clusters HFL for 32-bit floating point
python federated-hierarchical4_main.py --local_ep=5 --local_bs=50 --frac=0.1 --Cepochs=10 --model=cnn --dataset=cifar --iid=1 --gpu=1 --lr=0.01 --epochs=100 --num_cluster=4


# This is for 8 clusters HFL for 32-bit floating point
python federated-hierarchical8_main.py --local_ep=5 --local_bs=50 --Cepochs=10 --model=cnn --dataset=cifar --iid=1 --gpu=1 --lr=0.01 --epochs=100 --num_cluster=8



# ================ 16-bit ================ 
# This is for FL for 16-bit floating point
python ./federated_main_fp16.py --local_ep=5 --local_bs=50 --frac=0.1 --model=cnn --dataset=cifar --iid=1 --gpu=1 --gpu_id=$GPU_ID --lr=0.01 --epochs=300 


# This is for 2 clusters FL for 16-bit floating point
python ./federated-hierarchical2_main_fp16.py --local_ep=5 --local_bs=50 --frac=0.1 --Cepochs=10 --model=cnn --dataset=cifar --iid=1 --num_cluster=2 --gpu=1 --gpu_id=$GPU_ID --lr=0.01 --epochs=100 


# This is for 4 clusters FL for 16-bit floating point
python ./federated-hierarchical4_main_fp16.py --local_ep=5 --local_bs=50 --frac=0.1 --Cepochs=10 --model=cnn --dataset=cifar --iid=1 --gpu=1 --gpu_id=$GPU_ID --lr=0.01 --epochs=100 --num_cluster=4 


# This is for 8 clusters FL for 16-bit floating point
python ./federated-hierarchical8_main_fp16.py --local_ep=5 --local_bs=50 --Cepochs=10 --model=cnn --dataset=cifar --iid=1 --gpu=1 --gpu_id=$GPU_ID --lr=0.01 --epochs=100 --num_cluster=8 

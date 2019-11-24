#!/bin/bash
# Comments line start with a #
# Commands are surrounde by ()
# Website on how to write bash script https://hackernoon.com/know-shell-scripting-202b2fbe03a8

# Set GPU device
GPU_ID="cuda:0"


# ================ 32-bit ================ 
# This is for FL for 32-bit floating point
# IID
python federated_main.py --local_ep=1 --local_bs=10 --frac=0.1 --model=cnn --dataset=mnist --iid=1 --gpu=1 --lr=0.01 --test_acc=97 --epochs=100
# NON-IID
python federated_main.py --local_ep=1 --local_bs=10 --frac=0.1 --model=cnn --dataset=mnist --iid=0 --gpu=1 --lr=0.01 --epochs=300 --test_acc=97


# This is for 2 clusters HFL for 32-bit floating point
# IID
python federated-hierarchical2_main.py --local_ep=1 --local_bs=10 --frac=0.1 --Cepochs=10 --model=cnn --dataset=mnist --iid=1 --num_cluster=2 --gpu=1 --lr=0.01 --epochs=100
# NON-IID
python federated-hierarchical2_main.py --local_ep=1 --local_bs=10 --frac=0.1 --Cepochs=10 --model=cnn --dataset=mnist --iid=0 --num_cluster=2 --gpu=1 --lr=0.01 --epochs=100


# This is for 4 clusters HFL for 32-bit floating point
# IID
python federated-hierarchical4_main.py --local_ep=1 --local_bs=10 --frac=0.1 --Cepochs=10 --model=cnn --dataset=mnist --iid=1 --gpu=1 --lr=0.01 --epochs=100  --num_cluster=4
# NON-IID
python federated-hierarchical4_main.py --local_ep=1 --local_bs=10 --frac=0.1 --Cepochs=10 --model=cnn --dataset=mnist --iid=0 --gpu=1 --lr=0.01 --epochs=100  --num_cluster=4


# This is for 8 clusters HFL for 32-bit floating point
# IID
python federated-hierarchical8_main.py --local_ep=1 --local_bs=10 --Cepochs=10 --model=cnn --dataset=mnist --iid=1 --gpu=1 --lr=0.01 --epochs=30 --num_cluster=8 --test_acc=97
# NON-IID
python federated-hierarchical8_main.py --local_ep=1 --local_bs=10 --Cepochs=10 --model=cnn --dataset=mnist --iid=0 --gpu=1 --lr=0.01 --epochs=30 --num_cluster=8 --test_acc=97




# ================ 16-bit ================ 
# This is for 1 cluster FL for 16-bit floating point
python ./federated_main_fp16.py --local_ep=1 --local_bs=10 --frac=0.1 --model=cnn --dataset=mnist --iid=1 --gpu=1 --gpu_id=$GPU_ID --lr=0.01 --test_acc=97 --epochs=100 

python ./federated_main_fp16.py --local_ep=1 --local_bs=10 --frac=0.1 --model=cnn --dataset=mnist --iid=0 --gpu=1 --gpu_id=$GPU_ID --lr=0.01 --epochs=261 --test_acc=97 


# This is for 2 clusters FL for 16-bit floating point
python ./federated-hierarchical2_main_fp16.py --local_ep=1 --local_bs=10 --frac=0.1 --Cepochs=10 --model=cnn --dataset=mnist --iid=1 --num_cluster=2 --gpu=1 --gpu_id=$GPU_ID --lr=0.01 --epochs=100 

python ./federated-hierarchical2_main_fp16.py --local_ep=1 --local_bs=10 --frac=0.1 --Cepochs=10 --model=cnn --dataset=mnist --iid=0 --num_cluster=2 --gpu=1 --gpu_id=$GPU_ID --lr=0.01 --epochs=100 


# This is for 4 clusters FL for 16-bit floating point
python ./federated-hierarchical4_main_fp16.py --local_ep=1 --local_bs=10 --frac=0.1 --Cepochs=10 --model=cnn --dataset=mnist --iid=1 --gpu=1 --gpu_id=$GPU_ID --lr=0.01 --epochs=100  --num_cluster=4

python ./federated-hierarchical4_main_fp16.py --local_ep=1 --local_bs=10 --frac=0.1 --Cepochs=10 --model=cnn --dataset=mnist --iid=0 --gpu=1 --gpu_id=$GPU_ID --lr=0.01 --epochs=100  --num_cluster=4 

# This is for 8 clusters FL for 16-bit floating point
python ./federated-hierarchical8_main_fp16.py --local_ep=1 --local_bs=10 --frac=0.1 --Cepochs=10 --model=cnn --dataset=mnist --iid=1 --gpu=1 --gpu_id=$GPU_ID --lr=0.01 --epochs=30  --num_cluster=8 

python ./federated-hierarchical8_main_fp16.py --local_ep=1 --local_bs=10 --frac=0.1 --Cepochs=10 --model=cnn --dataset=mnist --iid=0 --gpu=1 --gpu_id=$GPU_ID --lr=0.01 --epochs=30  --num_cluster=8 

#!/bin/bash
# Comments line start with a #
# Commands are surrounde by ()
# Website on how to write bash script https://hackernoon.com/know-shell-scripting-202b2fbe03a8

# Set GPU device
GPU_ID="cuda:1"



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
# This is the baseline without FL for 16-bit floating point.
python ./baseline_main_fp16.py --epochs=10 --model=cnn --dataset=cifar --num_classes=10 --gpu=1 --gpu_id=$GPU_ID | tee -a ../logs/terminaloutput_cifar_fp16_baseline.txt &


# This is for 1 cluster FL for 16-bit floating point
python ./federated_main_fp16.py --local_ep=5 --local_bs=50 --frac=0.1 --model=cnn --dataset=cifar --iid=1 --gpu=1 --gpu_id=$GPU_ID --lr=0.01 --test_acc=85 --epochs=100 | tee -a ../logs/terminaloutput_cifar_fp16_1c_10ep_ta85.txt &

python ./federated_main_fp16.py --local_ep=5 --local_bs=50 --frac=0.1 --model=cnn --dataset=cifar --iid=1 --gpu=1 --gpu_id=$GPU_ID --lr=0.01 --epochs=200 | tee -a ../logs/terminaloutput_cifar_fp16_1c_200ep_ta95.txt &

python ./federated_main_fp16.py --local_ep=5 --local_bs=50 --frac=0.1 --model=cnn --dataset=cifar --iid=1 --gpu=1 --gpu_id=$GPU_ID --lr=0.01 --epochs=300 | tee -a ../logs/terminaloutput_cifar_fp16_1c_300ep_ta95.txt &


# This is for 2 clusters FL for 16-bit floating point
python ./federated-hierarchical2_main_fp16.py --local_ep=5 --local_bs=50 --frac=0.1 --Cepochs=10 --model=cnn --dataset=cifar --iid=1 --num_cluster=2 --gpu=1 --gpu_id=$GPU_ID --lr=0.01 --epochs=100 --test_acc=85 | tee -a ../logs/terminaloutput_cifar_fp16_2c_100ep_ta85.txt &

python ./federated-hierarchical2_main_fp16.py --local_ep=5 --local_bs=50 --frac=0.1 --Cepochs=10 --model=cnn --dataset=cifar --iid=1 --num_cluster=2 --gpu=1 --gpu_id=$GPU_ID --lr=0.01 --epochs=100 | tee -a ../logs/terminaloutput_cifar_fp16_2c_100ep_t95.txt &


# This is for 4 clusters FL for 16-bit floating point
python ./federated-hierarchical4_main_fp16.py --local_ep=5 --local_bs=50 --frac=0.1 --Cepochs=10 --model=cnn --dataset=cifar --iid=1 --gpu=1 --gpu_id=$GPU_ID --lr=0.01 --epochs=100 --num_cluster=4 | tee -a ../logs/terminaloutput_cifar_fp16_4c_100ep_t95.txt &


# This is for 8 clusters FL for 16-bit floating point
python ./federated-hierarchical8_main_fp16.py --local_ep=5 --local_bs=50 --Cepochs=10 --model=cnn --dataset=cifar --iid=1 --gpu=1 --gpu_id=$GPU_ID --lr=0.01 --epochs=100 --num_cluster=8 | tee -a ../logs/terminaloutput_cifar_fp16_8c_100ep_t95.txt &

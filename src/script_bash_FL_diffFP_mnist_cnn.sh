#!/bin/bash
# Comments line start with a #
# Commands are surrounde by ()
# Website on how to write bash script https://hackernoon.com/know-shell-scripting-202b2fbe03a8

# Set GPU device
GPU_ID="cuda:1"

# This is the baseline without FL for 16-bit floating point.
python ./baseline_main_fp16.py --epochs=10 --model=cnn --dataset=mnist --num_classes=10 --gpu=1  --gpu_id=$GPU_ID | tee -a ../logs/terminaloutput_mnist_CNN_fp16_baseline.txt &


# This is for 1 cluster FL for 16-bit floating point
python ./federated_main_fp16.py --local_ep=1 --local_bs=10 --frac=0.1 --model=cnn --dataset=mnist --iid=1 --gpu=1 --gpu_id=$GPU_ID --lr=0.01 --test_acc=97 --epochs=100 | tee -a ../logs/terminaloutput_mnist_CNN_fp16_1c1.txt &

python ./federated_main_fp16.py --local_ep=1 --local_bs=10 --frac=0.1 --model=cnn --dataset=mnist --iid=0 --gpu=1 --gpu_id=$GPU_ID --lr=0.01 --epochs=100 --test_acc=97 | tee -a ../logs/terminaloutput_mnist_CNN_fp16_1c2.txt &

python ./federated_main_fp16.py --local_ep=1 --local_bs=10 --frac=0.1 --model=cnn --dataset=mnist --iid=0 --gpu=1 --gpu_id=$GPU_ID --lr=0.01 --epochs=261 --test_acc=97 | tee -a ../logs/terminaloutput_mnist_CNN_fp16_1c3.txt &


# This is for 2 clusters FL for 16-bit floating point
python ./federated-hierarchical2_main_fp16.py --local_ep=1 --local_bs=10 --frac=0.1 --Cepochs=10 --model=cnn --dataset=mnist --iid=1 --num_cluster=2 --gpu=1 --gpu_id=$GPU_ID --lr=0.01 --epochs=100 | tee -a ../logs/terminaloutput_mnist_CNN_fp16_2c1.txt &

python ./federated-hierarchical2_main_fp16.py --local_ep=1 --local_bs=10 --frac=0.1 --Cepochs=10 --model=cnn --dataset=mnist --iid=0 --num_cluster=2 --gpu=1 --gpu_id=$GPU_ID --lr=0.01 --epochs=100 | tee -a ../logs/terminaloutput_mnist_CNN_fp16_2c2.txt &




# This is for 4 clusters FL for 16-bit floating point
python ./federated-hierarchical4_main_fp16.py --local_ep=1 --local_bs=10 --frac=0.1 --Cepochs=10 --model=cnn --dataset=mnist --iid=1 --gpu=1 --gpu_id=$GPU_ID --lr=0.01 --epochs=100  --num_cluster=4 | tee -a ../logs/terminaloutput_mnist_CNN_fp16_4c1.txt &

python ./federated-hierarchical4_main_fp16.py --local_ep=1 --local_bs=10 --frac=0.1 --Cepochs=10 --model=cnn --dataset=mnist --iid=0 --gpu=1 --gpu_id=$GPU_ID --lr=0.01 --epochs=100  --num_cluster=4 | tee -a ../logs/terminaloutput_mnist_CNN_fp16_4c2.txt &

# This is for 8 clusters FL for 16-bit floating point
python ./federated-hierarchical8_main_fp16.py --local_ep=1 --local_bs=10 --frac=0.1 --Cepochs=10 --model=cnn --dataset=mnist --iid=1 --gpu=1 --gpu_id=$GPU_ID --lr=0.01 --epochs=30  --num_cluster=8 | tee -a ../logs/terminaloutput_mnist_CNN_fp16_8c1.txt &



#!/bin/bash
# Comments line start with a #
# Commands are surrounde by ()
# Website on how to write bash script https://hackernoon.com/know-shell-scripting-202b2fbe03a8

# Set GPU device
GPU_ID="cuda:1"

# ================ 32-bit ================ 
# This is for FL for 32-bit floating point
# IID
python federated_main.py --local_ep=1 --local_bs=10 --frac=0.1 --model=mlp --dataset=mnist --iid=1 --gpu=1 --lr=0.01 --test_acc=95 --mlpdim=200 --epochs=200
# NON-IID
python federated_main.py --local_ep=1 --local_bs=10 --frac=0.1 --model=mlp --dataset=mnist --iid=0 --gpu=1 --lr=0.1 --test_acc=95 --mlpdim=200 --epochs=300


# This is for 2 clusters HFL for 32-bit floating point
# IID
python federated-hierarchical2_main.py --local_ep=1 --local_bs=10 --frac=0.1 --Cepochs=10 --model=mlp --dataset=mnist --iid=1 --num_cluster=2 --gpu=1 --lr=0.01 --mlpdim=200 --epochs=100
# NON-IID
python federated-hierarchical2_main.py --local_ep=1 --local_bs=10 --frac=0.1 --Cepochs=10 --model=mlp --dataset=mnist --iid=0 --num_cluster=2 --gpu=1 --lr=0.01 --mlpdim=200 --epochs=100


# This is for 4 clusters HFL for 32-bit floating point
# IID
python federated-hierarchical4_main.py --local_ep=1 --local_bs=10 --frac=0.1 --Cepochs=10 --model=mlp --dataset=mnist --iid=1 --gpu=1 --lr=0.01 --mlpdim=200 --epochs=100  --num_cluster=4
# NON-IID
python federated-hierarchical4_main.py --local_ep=1 --local_bs=10 --frac=0.1 --Cepochs=10 --model=mlp --dataset=mnist --iid=0 --gpu=1 --lr=0.01 --mlpdim=200 --epochs=150  --num_cluster=4


# This is for 8 clusters HFL for 32-bit floating point
# IID
python federated-hierarchical8_main.py --local_ep=1 --local_bs=10 --Cepochs=10 --model=mlp --dataset=mnist --iid=1 --gpu=1 --lr=0.01 --mlpdim=200 --epochs=30 --num_cluster=8 --test_acc=95
# NON-IID
python federated-hierarchical8_main.py --local_ep=1 --local_bs=10 --Cepochs=10 --model=mlp --dataset=mnist --iid=0 --gpu=1 --lr=0.01 --mlpdim=200 --epochs=30 --num_cluster=8 --test_acc=95




# ================ 16-bit ================ 
# This is the baseline without FL for 16-bit floating point.
python ./baseline_main_fp16.py --epochs=10 --model=mlp --dataset=mnist --num_classes=10 --gpu=1 --gpu_id=$GPU_ID | tee -a ../logs/terminaloutput_mnist_fp16_baseline.txt &


# This is for 1 cluster FL for 16-bit floating point
python ./federated_main_fp16.py --local_ep=1 --local_bs=10 --frac=0.1 --model=mlp --dataset=mnist --iid=1 --gpu=1 --gpu_id=$GPU_ID --lr=0.01 --test_acc=95 --mlpdim=200 --epochs=200 | tee -a ../logs/terminaloutput_mnist_fp16_1c.txt &

python ./federated_main_fp16.py --local_ep=1 --local_bs=10 --frac=0.1 --model=mlp --dataset=mnist --iid=0 --gpu=1 --gpu_id=$GPU_ID --lr=0.1 --test_acc=95 --mlpdim=200 --epochs=300 | tee -a ../logs/terminaloutput_mnist_fp16_1c.txt &

python ./federated_main_fp16.py --local_ep=1 --local_bs=10 --frac=0.1 --model=mlp --dataset=mnist --iid=1 --gpu=1 --gpu_id=$GPU_ID --lr=0.1 --test_acc=95 --mlpdim=250 --epochs=200 | tee -a ../logs/terminaloutput_mnist_fp16_1c.txt &

# FL_mnist_mlp_468_C[0.1]_iid[1]_E[1]_B[10]
python ./federated_main_fp16.py --local_ep=1 --local_bs=10 --frac=0.1 --model=mlp --dataset=mnist --iid=1 --gpu=1 --gpu_id=$GPU_ID --lr=0.01 --test_acc=95 --mlpdim=200 --epochs=468 | tee -a ../logs/terminaloutput_mnist_fp16_1c_468epoch.txt &

# FL_mnist_mlp_1196_lr[0.01]_C[0.1]_iid[0]_E[1]_B[10]
python ./federated_main_fp16.py --local_ep=1 --local_bs=10 --frac=0.1 --model=mlp --dataset=mnist --iid=0 --gpu=1 --gpu_id=$GPU_ID --lr=0.01 --test_acc=95 --mlpdim=200 --epochs=1196 | tee -a ../logs/terminaloutput_mnist_fp16_1c_1196epoch.txt &


# This is for 2 clusters FL for 16-bit floating point
python ./federated-hierarchical2_main_fp16.py --local_ep=1 --local_bs=10 --frac=0.1 --Cepochs=10 --model=mlp --dataset=mnist --iid=1 --num_cluster=2 --gpu=1 --gpu_id=$GPU_ID --lr=0.01 --mlpdim=200 --epochs=100 --test_acc=94 | tee -a ../logs/terminaloutput_mnist_fp16_2c.txt &

python ./federated-hierarchical2_main_fp16.py --local_ep=1 --local_bs=10 --frac=0.1 --Cepochs=10 --model=mlp --dataset=mnist --iid=0 --num_cluster=2 --gpu=1 --gpu_id=$GPU_ID --lr=0.05 --mlpdim=200 --epochs=100 --test_acc=94 | tee -a ../logs/terminaloutput_mnist_fp16_2c.txt &

python ./federated-hierarchical2_main_fp16.py --local_ep=1 --local_bs=10 --frac=0.1 --Cepochs=10 --model=mlp --dataset=mnist --iid=1 --num_cluster=2 --gpu=1 --gpu_id=$GPU_ID --lr=0.01 --mlpdim=200 --epochs=100 | tee -a ../logs/terminaloutput_mnist_fp16_2c.txt &

python ./federated-hierarchical2_main_fp16.py --local_ep=1 --local_bs=10 --frac=0.1 --Cepochs=10 --model=mlp --dataset=mnist --iid=0 --num_cluster=2 --gpu=1 --gpu_id=$GPU_ID --lr=0.01 --mlpdim=200 --epochs=100 | tee -a ../logs/terminaloutput_mnist_fp16_2c.txt &




# This is for 4 clusters FL for 16-bit floating point
python ./federated-hierarchical4_main_fp16.py --local_ep=1 --local_bs=10 --frac=0.1 --Cepochs=10 --model=mlp --dataset=mnist --iid=1 --num_cluster=4 --gpu=1 --gpu_id=$GPU_ID --lr=0.1 --mlpdim=200 --epochs=100 --test_acc=95 | tee -a ../logs/terminaloutput_mnist_fp16_4c.txt

python ./federated-hierarchical4_main_fp16.py --local_ep=1 --local_bs=10 --frac=0.1 --Cepochs=10 --model=mlp --dataset=mnist --iid=1 --num_cluster=4 --gpu=1 --gpu_id=$GPU_ID --lr=0.05 --mlpdim=200 --epochs=100 | tee -a ../logs/terminaloutput_mnist_fp16_4c.txt

python ./federated-hierarchical4_main_fp16.py --local_ep=1 --local_bs=10 --frac=0.1 --Cepochs=10 --model=mlp --dataset=mnist --iid=0 --num_cluster=4 --gpu=1 --gpu_id=$GPU_ID --lr=0.05 --mlpdim=200 --epochs=100 | tee -a ../logs/terminaloutput_mnist_fp16_4c.txt

python ./federated-hierarchical4_main_fp16.py --local_ep=1 --local_bs=10 --frac=0.1 --Cepochs=10 --model=mlp --dataset=mnist --iid=1 --num_cluster=4 --gpu=1 --gpu_id=$GPU_ID --lr=0.01 --mlpdim=200 --epochs=150 | tee -a ../logs/terminaloutput_mnist_fp16_4c.txt

python ./federated-hierarchical4_main_fp16.py --local_ep=1 --local_bs=10 --frac=0.1 --Cepochs=10 --model=mlp --dataset=mnist --iid=0 --num_cluster=4 --gpu=1 --gpu_id=$GPU_ID --lr=0.05 --mlpdim=200 --epochs=150 | tee -a ../logs/terminaloutput_mnist_fp16_4c.txt

python ./federated-hierarchical4_main_fp16.py --local_ep=1 --local_bs=10 --frac=0.1 --Cepochs=10 --model=mlp --dataset=mnist --iid=1 --num_cluster=4 --gpu=1 --gpu_id=$GPU_ID --lr=0.01 --mlpdim=200 --epochs=150 --optimizer='adam' | tee -a ../logs/terminaloutput_mnist_fp16_4c.txt

python ./federated-hierarchical4_main_fp16.py --local_ep=1 --local_bs=10 --frac=0.1 --Cepochs=10 --model=mlp --dataset=mnist --iid=1 --gpu=1 --gpu_id=$GPU_ID --lr=0.01 --mlpdim=200 --epochs=100 | tee -a ../logs/terminaloutput_mnist_fp16_4c.txt

python ./federated-hierarchical4_main_fp16.py --local_ep=1 --local_bs=10 --frac=0.1 --Cepochs=10 --model=mlp --dataset=mnist --iid=1 --gpu=1 --gpu_id=$GPU_ID --lr=0.01 --mlpdim=200 --epochs=100 | tee -a ../logs/terminaloutput_mnist_fp16_4c.txt

python ./federated-hierarchical4_main_fp16.py --local_ep=1 --local_bs=10 --frac=0.1 --Cepochs=10 --model=mlp --dataset=mnist --iid=0 --gpu=1 --gpu_id=$GPU_ID --lr=0.01 --mlpdim=200 --epochs=100 | tee -a ../logs/terminaloutput_mnist_fp16_4c.txt

python ./federated-hierarchical4_main_fp16.py --local_ep=1 --local_bs=10 --frac=0.1 --Cepochs=10 --model=mlp --dataset=mnist --iid=1 --gpu=1 --gpu_id=$GPU_ID --lr=0.01 --mlpdim=200 --epochs=100  --num_cluster=4 | tee -a ../logs/terminaloutput_mnist_fp16_4c.txt

python ./federated-hierarchical4_main_fp16.py --local_ep=1 --local_bs=10 --frac=0.1 --Cepochs=10 --model=mlp --dataset=mnist --iid=0 --gpu=1 --gpu_id=$GPU_ID --lr=0.01 --mlpdim=200 --epochs=150  --num_cluster=4 | tee -a ../logs/terminaloutput_mnist_fp16_4c.txt

# HFL4_mnist_mlp_30_lr[0.01]_C[0.1]_iid[1]_E[1]_B[10]
python ./federated-hierarchical4_main_fp16.py --local_ep=1 --local_bs=10 --frac=0.1 --Cepochs=10 --model=mlp --dataset=mnist --iid=1 --num_cluster=4 --gpu=1 --gpu_id=$GPU_ID --lr=0.01 --mlpdim=200 --epochs=30 --test_acc=95 | tee -a ../logs/terminaloutput_mnist_fp16_4c_30epoch.txt &


# This is for 8 clusters FL for 16-bit floating point
python ./federated-hierarchical8_main_fp16.py --local_ep=1 --local_bs=10 --Cepochs=10 --model=mlp --dataset=mnist --iid=1 --gpu=1 --gpu_id=$GPU_ID --lr=0.01 --mlpdim=200 --epochs=30 --num_cluster=8 --test_acc=95 | tee -a ../logs/terminaloutput_mnist_fp16_8c.txt

python ./federated-hierarchical8_main_fp16.py --local_ep=1 --local_bs=10 --Cepochs=10 --model=mlp --dataset=mnist --iid=0 --gpu=1 --gpu_id=$GPU_ID --lr=0.01 --mlpdim=200 --epochs=30 --num_cluster=8 --test_acc=95 | tee -a ../logs/terminaloutput_mnist_fp16_8c.txt


#!/bin/bash
# Comments line start with a #
# Commands are surrounde by ()
# Website on how to write bash script https://hackernoon.com/know-shell-scripting-202b2fbe03a8

# This is the baseline without FL for 16-bit floating point.
python ./baseline_main_fp16.py --epochs=10 --model="mlp" --dataset="mnist" --num_classes=10 --gpu=1 --gpu_id="cuda:0" --mlpdim=200

python ./federated_main_fp16.py --local_ep=1 --local_bs=10 --frac=0.1 --model=mlp --dataset=mnist --iid=1 --gpu=1 --lr=0.01 --test_acc=95 --mlpdim=200 --epochs=200

python ./federated-hierarchical2_main_fp16.py --local_ep=1 --local_bs=10 --frac=0.1 --Cepochs=10 --model=mlp --dataset=mnist --iid=1 --num_cluster=2 --gpu=1 --lr=0.01 --mlpdim=200 --epochs=100 --test_acc=94

python ./federated-hierarchical4_main_fp16.py --local_ep=1 --local_bs=10 --frac=0.1 --Cepochs=10 --model=mlp --dataset=mnist --iid=1 --num_cluster=4 --gpu=1 --lr=0.1 --mlpdim=200 --epochs=100 --test_acc=95

python ./federated-hierarchical8_main_fp16.py --local_ep=1 --local_bs=10 --Cepochs=10 --model=mlp --dataset=mnist --iid=1 --gpu=1 --lr=0.01 --mlpdim=200 --epochs=30 --num_cluster=8 --test_acc=95

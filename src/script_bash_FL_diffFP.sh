#!/bin/bash
# Comments line start with a #
# Commands are surrounde by ()
# Website on how to write bash script https://hackernoon.com/know-shell-scripting-202b2fbe03a8

# This is the baseline without FL for 16-bit floating point.
python ./baseline_main.py --floating_point_16 --epochs=10 --model="mlp" --dataset="mnist" --num_classes=10 --gpu --gpu_id="cuda:0" --mlpdim=200

python ./federated_main.py --floating_point_16 --local_ep=1 --local_bs=10 --frac=0.1 --model=mlp --dataset=mnist --iid=1 --gpu --learning_rate=0.01 --test_acc=95 --mlpdim=200 --epochs=200

python ./federated-hierarchical_main.py --floating_point_16 --local_ep=1 --local_bs=10 --frac=0.1 --Cepochs=10 --model=mlp --dataset=mnist --iid=1 --num_cluster=2 --gpu --learning_rate=0.01 --mlpdim=200 --epochs=100 --test_acc=94

python ./federated-hierarchical_main.py --floating_point_16 --local_ep=1 --local_bs=10 --frac=0.1 --Cepochs=10 --model=mlp --dataset=mnist --iid=1 --num_cluster=4 --gpu --learning_rate=0.1 --mlpdim=200 --epochs=100 --test_acc=95

python ./federated-hierarchical_main.py --floating_point_16 --local_ep=1 --local_bs=10 --Cepochs=10 --model=mlp --dataset=mnist --iid=1 --gpu --learning_rate=0.01 --mlpdim=200 --epochs=30 --num_cluster=8 --test_acc=95

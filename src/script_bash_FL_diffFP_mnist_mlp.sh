#!/bin/bash
# Comments line start with a #
# Commands are surrounde by ()
# Website on how to write bash script https://hackernoon.com/know-shell-scripting-202b2fbe03a8

# Set GPU device
GPU_ID="cuda:0"

# ================ 32-bit ================
# This is for FL for 32-bit floating point
# IID
python federated_main.py --local_ep=1 --local_bs=10 --frac=0.1 --model=mlp --dataset=mnist --iid=1 --gpu --lr=0.01 --test_acc=95 --mlpdim=200 --epochs=600
# NON-IID
python federated_main.py --local_ep=1 --local_bs=10 --frac=0.1 --model=mlp --dataset=mnist --iid=0 --gpu --lr=0.1 --test_acc=95 --mlpdim=200 --epochs=1200

python federated_main.py --local_ep=1 --local_bs=10 --frac=0.1 --model=mlp --dataset=mnist --iid=0 --gpu --lr=0.1 --test_acc=95 --mlpdim=200 --epochs=1196

# This is for 2 clusters HFL for 32-bit floating point
# IID
python federated-hierarchical_main.py --local_ep=1 --local_bs=10 --frac=0.1 --Cepochs=10 --model=mlp --dataset=mnist --iid=1 --num_cluster=2 --gpu --lr=0.01 --mlpdim=200 --epochs=100
# NON-IID
python federated-hierarchical_main.py --local_ep=1 --local_bs=10 --frac=0.1 --Cepochs=10 --model=mlp --dataset=mnist --iid=0 --num_cluster=2 --gpu --lr=0.01 --mlpdim=200 --epochs=100


# This is for 4 clusters HFL for 32-bit floating point
# IID
python federated-hierarchical_main.py --local_ep=1 --local_bs=10 --frac=0.1 --Cepochs=10 --model=mlp --dataset=mnist --iid=1 --gpu --lr=0.01 --mlpdim=200 --epochs=100  --num_cluster=4
# NON-IID
python federated-hierarchical_main.py --local_ep=1 --local_bs=10 --frac=0.1 --Cepochs=10 --model=mlp --dataset=mnist --iid=0 --gpu --lr=0.01 --mlpdim=200 --epochs=150  --num_cluster=4


# This is for 8 clusters HFL for 32-bit floating point
# IID
python federated-hierarchical_main.py --local_ep=1 --local_bs=10 --Cepochs=10 --model=mlp --dataset=mnist --iid=1 --gpu --lr=0.01 --mlpdim=200 --epochs=30 --num_cluster=8 --test_acc=95
# NON-IID
python federated-hierarchical_main.py --local_ep=1 --local_bs=10 --Cepochs=10 --model=mlp --dataset=mnist --iid=0 --gpu --lr=0.01 --mlpdim=200 --epochs=30 --num_cluster=8 --test_acc=95



# ================ 16-bit ================
# This is for 1 cluster FL for 16-bit floating point
python ./federated_main.py --floating_point_16 --local_ep=1 --local_bs=10 --frac=0.1 --model=mlp --dataset=mnist --iid=1 --gpu --gpu_id=$GPU_ID --lr=0.01 --test_acc=95 --mlpdim=200 --epochs=650

python ./federated_main.py --floating_point_16 --local_ep=1 --local_bs=10 --frac=0.1 --model=mlp --dataset=mnist --iid=1 --gpu --gpu_id=$GPU_ID --lr=0.01 --test_acc=95 --mlpdim=200 --epochs=468

python ./federated_main.py --floating_point_16 --local_ep=1 --local_bs=10 --frac=0.1 --model=mlp --dataset=mnist --iid=0 --gpu --gpu_id=$GPU_ID --lr=0.1 --test_acc=95 --mlpdim=200 --epochs=1196


# This is for 2 clusters FL for 16-bit floating point
python ./federated-hierarchical_main.py --floating_point_16 --local_ep=1 --local_bs=10 --frac=0.1 --Cepochs=10 --model=mlp --dataset=mnist --iid=1 --num_cluster=2 --gpu --gpu_id=$GPU_ID --lr=0.01 --mlpdim=200 --epochs=100

python ./federated-hierarchical_main.py --floating_point_16 --local_ep=1 --local_bs=10 --frac=0.1 --Cepochs=10 --model=mlp --dataset=mnist --iid=0 --num_cluster=2 --gpu --gpu_id=$GPU_ID --lr=0.01 --mlpdim=200 --epochs=100


# This is for 4 clusters FL for 16-bit floating point
python ./federated-hierarchical_main.py --floating_point_16 --local_ep=1 --local_bs=10 --frac=0.1 --Cepochs=10 --model=mlp --dataset=mnist --iid=1 --num_cluster=4 --gpu --gpu_id=$GPU_ID --lr=0.01 --mlpdim=200 --epochs=100

python ./federated-hierarchical_main.py --floating_point_16 --local_ep=1 --local_bs=10 --frac=0.1 --Cepochs=10 --model=mlp --dataset=mnist --iid=0 --num_cluster=4 --gpu --gpu_id=$GPU_ID --lr=0.01 --mlpdim=200 --epochs=150


# This is for 8 clusters FL for 16-bit floating point
python ./federated-hierarchical_main.py --floating_point_16 --local_ep=1 --local_bs=10 --Cepochs=10 --model=mlp --dataset=mnist --iid=1 --gpu --gpu_id=$GPU_ID --lr=0.01 --mlpdim=200 --epochs=30 --num_cluster=8 --test_acc=95

python ./federated-hierarchical_main.py --floating_point_16 --local_ep=1 --local_bs=10 --Cepochs=10 --model=mlp --dataset=mnist --iid=0 --gpu --gpu_id=$GPU_ID --lr=0.01 --mlpdim=200 --epochs=30 --num_cluster=8 --test_acc=95

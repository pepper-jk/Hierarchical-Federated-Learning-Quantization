#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details, set_device, build_model, fl_train
import math
import random


if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    # Select CPU or GPU
    device = set_device(args)

    # load dataset and user groups
    train_dataset, test_dataset, user_groupsold = get_dataset(args)

    # user_groups = user_groupsold
    # keylist = list(user_groups.keys())
    # ======= Shuffle dataset =======
    keys =  list(user_groupsold.keys())
    random.shuffle(keys)
    user_groups = dict()
    for key in keys:
        user_groups.update({key:user_groupsold[key]})
    # print(user_groups.keys())
    keylist = list(user_groups.keys())
    print("keylist: ", keylist)
    # ======= Splitting into clusters. FL groups =======
    if args.num_clusters != 8:
        exit("Confirm that the number of clusters is 8?")
    cluster_size = int(args.num_users / args.num_clusters)
    print("Each cluster size: ", cluster_size)

    # Cluster 1
    A1 = keylist[:cluster_size]
    # A1 = np.random.choice(keylist, cluster_size, replace=False)
    print("A1: ", A1)
    user_groupsA = {k:user_groups[k] for k in A1 if k in user_groups}
    print("Size of cluster 1: ", len(user_groupsA))
    # Cluster 2
    B1 = keylist[cluster_size:2*cluster_size]
    # B1 = np.random.choice(keylist, cluster_size, replace=False)
    print("B1: ", B1)
    user_groupsB = {k:user_groups[k] for k in B1 if k in user_groups}
    print("Size of cluster 2: ", len(user_groupsB))
    # Cluster 3
    C1 = keylist[2*cluster_size:3*cluster_size]
    # C1 = np.random.choice(keylist, cluster_size, replace=False)
    print("C1: ", C1)
    user_groupsC = {k:user_groups[k] for k in C1 if k in user_groups}
    print("Size of cluster 3: ", len(user_groupsC))
    # Cluster 4
    D1 = keylist[3*cluster_size:4*cluster_size]
    # D1 = np.random.choice(keylist, cluster_size, replace=False)
    print("D1: ", D1)
    user_groupsD = {k:user_groups[k] for k in D1 if k in user_groups}
    print("Size of cluster 4: ", len(user_groupsD))
    # Cluster 5
    E1 = keylist[4*cluster_size:5*cluster_size] #np.random.choice(keylist, cluster_size, replace=False)
    print("E1: ", E1)
    user_groupsE = {k:user_groups[k] for k in E1 if k in user_groups}
    print("Size of cluster 5: ", len(user_groupsE))
    # Cluster 6
    F1 = keylist[5*cluster_size:6*cluster_size] #np.random.choice(keylist, cluster_size, replace=False)
    print("F1: ", F1)
    user_groupsF = {k:user_groups[k] for k in F1 if k in user_groups}
    print("Size of cluster 6: ", len(user_groupsF))
    # Cluster 7
    G1 = keylist[6*cluster_size:7*cluster_size] #np.random.choice(keylist, cluster_size, replace=False)
    print("G1: ", G1)
    user_groupsG = {k:user_groups[k] for k in G1 if k in user_groups}
    print("Size of cluster 7: ", len(user_groupsC))
    # Cluster 8
    H1 = keylist[7*cluster_size:] #np.random.choice(keylist, cluster_size, replace=False)
    print("H1: ", H1)
    user_groupsH = {k:user_groups[k] for k in H1 if k in user_groups}
    print("Size of cluster 8: ", len(user_groupsH))

    # MODEL PARAM SUMMARY
    global_model = build_model(args, train_dataset)
    pytorch_total_params = sum(p.numel() for p in global_model.parameters())
    print("Model total number of parameters: ", pytorch_total_params)

    # from torchsummary import summary
    # summary(global_model, (1, 28, 28))
    # global_model.parameters()

    # Set the model to train and send it to device.
    global_model.to(device)
    # Set model to use Floating Point 16
    global_model.to(dtype=torch.float16)  ##########################
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()


    # ======= Set the cluster models to train and send it to device. =======
    # Cluster A
    cluster_modelA = build_model(args, train_dataset)
    cluster_modelA.to(device)
    cluster_modelA.to(dtype=torch.float16)
    cluster_modelA.train()
    # copy weights
    cluster_modelA_weights = cluster_modelA.state_dict()

    # Cluster B
    cluster_modelB = build_model(args, train_dataset)
    cluster_modelB.to(device)
    cluster_modelB.to(dtype=torch.float16)
    cluster_modelB.train()
    cluster_modelB_weights = cluster_modelB.state_dict()

    # Cluster C
    cluster_modelC = build_model(args, train_dataset)
    cluster_modelC.to(device)
    cluster_modelC.to(dtype=torch.float16)
    cluster_modelC.train()
    cluster_modelC_weights = cluster_modelC.state_dict()

    # Cluster D
    cluster_modelD = build_model(args, train_dataset)
    cluster_modelD.to(device)
    cluster_modelD.to(dtype=torch.float16)
    cluster_modelD.train()
    cluster_modelD_weights = cluster_modelD.state_dict()

    # Cluster E
    cluster_modelE = build_model(args, train_dataset)
    cluster_modelE.to(device)
    cluster_modelE.to(dtype=torch.float16)
    cluster_modelE.train()
    cluster_modelE_weights = cluster_modelE.state_dict()

    # Cluster F
    cluster_modelF = build_model(args, train_dataset)
    cluster_modelF.to(device)
    cluster_modelF.to(dtype=torch.float16)
    cluster_modelF.train()
    cluster_modelF_weights = cluster_modelF.state_dict()

    # Cluster G
    cluster_modelG = build_model(args, train_dataset)
    cluster_modelG.to(device)
    cluster_modelG.to(dtype=torch.float16)
    cluster_modelG.train()
    cluster_modelG_weights = cluster_modelG.state_dict()

    # Cluster H
    cluster_modelH = build_model(args, train_dataset)
    cluster_modelH.to(device)
    cluster_modelH.to(dtype=torch.float16)
    cluster_modelH.train()
    cluster_modelH_weights = cluster_modelH.state_dict()


    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 1
    val_loss_pre, counter = 0, 0
    testacc_check, epoch = 0, 0
    idx = np.random.randint(0,99)

    # for epoch in tqdm(range(args.epochs)):
    for epoch in range(args.epochs):
    # while testacc_check < args.test_acc or epoch < args.epochs:
    # while epoch < args.epochs:
        local_weights, local_losses, local_accuracies= [], [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        # ============== TRAIN ==============
        global_model.train()

        # ===== Cluster A =====
        A_model, A_weights, A_losses = fl_train(args, train_dataset, cluster_modelA, A1, user_groupsA, args.Cepochs, logger, cluster_dtype=torch.float16)
        local_weights.append(copy.deepcopy(A_weights))
        local_losses.append(copy.deepcopy(A_losses))
        cluster_modelA = global_model# = A_model
        # ===== Cluster B =====
        B_model, B_weights, B_losses = fl_train(args, train_dataset, cluster_modelB, B1, user_groupsB, args.Cepochs, logger, cluster_dtype=torch.float16)
        local_weights.append(copy.deepcopy(B_weights))
        local_losses.append(copy.deepcopy(B_losses))
        cluster_modelB = global_model# = B_model
        # ===== Cluster C =====
        C_model, C_weights, C_losses = fl_train(args, train_dataset, cluster_modelC, C1, user_groupsC, args.Cepochs, logger, cluster_dtype=torch.float16)
        local_weights.append(copy.deepcopy(C_weights))
        local_losses.append(copy.deepcopy(C_losses))
        cluster_modelC = global_model# = C_model
        # ===== Cluster D =====
        D_model, D_weights, D_losses = fl_train(args, train_dataset, cluster_modelD, D1, user_groupsD, args.Cepochs, logger, cluster_dtype=torch.float16)
        local_weights.append(copy.deepcopy(D_weights))
        local_losses.append(copy.deepcopy(D_losses))
        cluster_modelD = global_model# = D_model
        # ===== Cluster E =====
        E_model, E_weights, E_losses = fl_train(args, train_dataset, cluster_modelE, E1, user_groupsE, args.Cepochs, logger, cluster_dtype=torch.float16)
        local_weights.append(copy.deepcopy(E_weights))
        local_losses.append(copy.deepcopy(E_losses))
        cluster_modelE = global_model# = E_model
        # ===== Cluster F =====
        F_model, F_weights, F_losses = fl_train(args, train_dataset, cluster_modelF, F1, user_groupsF, args.Cepochs, logger, cluster_dtype=torch.float16)
        local_weights.append(copy.deepcopy(F_weights))
        local_losses.append(copy.deepcopy(F_losses))
        cluster_modelF = global_model# = F_model
        # ===== Cluster G =====
        G_model, G_weights, G_losses = fl_train(args, train_dataset, cluster_modelG, G1, user_groupsG, args.Cepochs, logger, cluster_dtype=torch.float16)
        local_weights.append(copy.deepcopy(G_weights))
        local_losses.append(copy.deepcopy(G_losses))
        cluster_modelG = global_model# = G_model
        # ===== Cluster H =====
        H_model, H_weights, H_losses = fl_train(args, train_dataset, cluster_modelH, H1, user_groupsH, args.Cepochs, logger, cluster_dtype=torch.float16)
        local_weights.append(copy.deepcopy(H_weights))
        local_losses.append(copy.deepcopy(H_losses))
        cluster_modelH = global_model# = H_model


        # averaging global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # ============== EVAL ==============
        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        # print("========== idx ========== ", idx)
        for c in range(args.num_users):
        # for c in range(cluster_size):
        # C = np.random.choice(keylist, int(args.frac * args.num_users), replace=False) # random set of clients
        # print("C: ", C)
        # for c in C:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[c], logger=logger)
            acc, loss = local_model.inference(model=global_model, dtype=torch.float16)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))
        # Add
        testacc_check = 100*train_accuracy[-1]
        epoch = epoch + 1

        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))


    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset, dtype=torch.float16)

    # print(f' \n Results after {args.epochs} global rounds of training:')
    print(f"\nAvg Training Stats after {epoch} global rounds:")
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    # Saving the objects train_loss and train_accuracy:
    file_name = '../save/objects_fp16/HFL8_{}_{}_{}_lr[{}]_C[{}]_iid[{}]_E[{}]_B[{}]_FP16.pkl'.\
    format(args.dataset, args.model, epoch, args.lr, args.frac, args.iid,
           args.local_ep, args.local_bs)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

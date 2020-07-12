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

    data_type = torch.float32
    appendage = ''
    # Set model to use Floating Point 16
    if args.floating_point_16:
        data_type = torch.float16
        appendage = '_FP16'

    # load dataset and user groups
    train_dataset, test_dataset, user_groupsold = get_dataset(args)

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
    cluster_size = int(args.num_users / args.num_clusters)
    print("Each cluster size: ", cluster_size)

    keylists_per_cluster = []
    user_groups_per_cluster = []

    for i in range(0, args.num_clusters):
        # Cluster i
        keylist_cluster = keylist[i*cluster_size:(i+1)*cluster_size]
        # TODO: make a cli argument cluster members random
        # FIXME: randomize only over left over keys
        # keylist_cluster = np.random.choice(keylist, cluster_size, replace=False)
        keylists_per_cluster.append(keylist_cluster)
        print("Cluster {}: ".format(i), keylist_cluster)
        user_groups_cluster = {k:user_groups[k] for k in keylist_cluster if k in user_groups}
        user_groups_per_cluster.append(user_groups_cluster)
        print("Size of cluster {}: ".format(i), len(user_groups_cluster))

    # MODEL PARAM SUMMARY
    global_model = build_model(args, train_dataset)
    pytorch_total_params = sum(p.numel() for p in global_model.parameters())
    print("Model total number of parameters: ", pytorch_total_params)

    # Set the model to train and send it to device.
    global_model.to(device, dtype=data_type)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # ======= Set the cluster models to train and send it to device. =======
    model_per_cluster = []
    weights_per_cluster = []

    for i in range(0, args.num_clusters):
        # build model
        cluster_model = build_model(args, train_dataset)
        cluster_model.to(device, dtype=data_type)
        cluster_model.train()
        # copy weights
        cluster_model_weights = cluster_model.state_dict()
        # save model and weights for later use
        model_per_cluster.append(cluster_model)
        weights_per_cluster.append(cluster_model_weights)

    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 1
    val_loss_pre, counter = 0, 0
    testacc_check, epoch = 0, 0

    for epoch in range(args.epochs):
        local_weights, local_losses, local_accuracies= [], [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        # ============== TRAIN ==============
        global_model.train()

        # ===== Clusters =====

        for i in range(0, args.num_clusters):
            model, weights, losses = fl_train(args, train_dataset, model_per_cluster[i], keylists_per_cluster[i], user_groups_per_cluster[i], args.Cepochs, logger, cluster_dtype=data_type)
            local_weights.append(copy.deepcopy(weights))
            local_losses.append(copy.deepcopy(losses))
            model_per_cluster[i] = global_model# = model

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
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[c], logger=logger)
            acc, loss = local_model.inference(model=global_model, dtype=data_type)
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
    test_acc, test_loss = test_inference(args, global_model, test_dataset, dtype=data_type)

    # print(f' \n Results after {args.epochs} global rounds of training:')
    print(f"\nAvg Training Stats after {epoch} global rounds:")
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    # Saving the objects train_loss and train_accuracy:
    file_name = '../save/objects{}/HFL2_{}_{}_{}_lr[{}]_C[{}]_iid[{}]_E[{}]_B[{}]{}.pkl'.\
    format(appendage.lower(), args.dataset, args.model, epoch, args.lr, args.frac,
           args.iid, args.local_ep, args.local_bs, appendage)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

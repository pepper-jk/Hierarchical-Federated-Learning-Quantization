#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os
import copy
import math
import numpy as np
import pickle
import random
import tensorboardX
import time
import torch
from tqdm import tqdm

import options
import output
import privacy_engine_xl as dp_xl
import update
import utils


if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = tensorboardX.SummaryWriter('../logs')

    args = options.args_parser()
    utils.exp_details(args)

    # get cli arguments
    floating_point_16 = args.floating_point_16
    epochs = args.epochs
    num_users = args.num_users

    ## cluster arguments
    Cepochs = args.Cepochs
    num_clusters = args.num_clusters

    # differential privacy
    sigma_local = args.sigma_local
    sigma_global = args.sigma_global
    sigma_intermediate = args.sigma_intermediate
    noise = args.noise

    ## for filesave
    model = args.model
    dataset = args.dataset
    learning_rate = args.learning_rate
    frac = args.frac
    iid = args.iid
    local_ep = args.local_ep
    local_bs = args.local_bs

    ## plots
    plot = args.plot

    # Select CPU or GPU
    device = device = utils.set_device(args)

    data_type = torch.float32
    appendage = ''
    # Set model to use Floating Point 16
    if floating_point_16:
        data_type = torch.float16
        appendage = '_FP16'

    # load dataset and user groups
    train_dataset, test_dataset, user_groupsold = utils.get_dataset(args)

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
    cluster_size = int(num_users / num_clusters)
    print("Each cluster size: ", cluster_size)

    keylists_per_cluster = []
    user_groups_per_cluster = []

    for i in range(0, num_clusters):
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
    global_model = utils.build_model(args, train_dataset)
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

    for i in range(0, num_clusters):
        # build model
        cluster_model = utils.build_model(args, train_dataset)
        cluster_model.to(device, dtype=data_type)
        cluster_model.train()
        # copy weights
        cluster_model_weights = cluster_model.state_dict()
        # save model and weights for later use
        model_per_cluster.append(cluster_model)
        weights_per_cluster.append(cluster_model_weights)

    train_loss, train_accuracy = [], []
    test_losses, test_accuracies = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 1
    val_loss_pre, counter = 0, 0
    testacc_check, epoch = 0, 0

    num_users_per_epoch = max(int(frac * num_users), 1)

    for epoch in range(epochs):
        local_weights, local_losses, local_accuracies= [], [], []
        cluster_percentage = []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        # ============== TRAIN ==============
        global_model.train()

        # ===== Clusters =====

        for i in range(0, num_clusters):
            cluster_model, weights, losses = utils.fl_train(args, train_dataset, model_per_cluster[i], keylists_per_cluster[i], user_groups_per_cluster[i], Cepochs, logger,
                                                            local_bs, device, sigma_intermediate, noise, num_users_per_epoch, cluster_dtype=data_type)
            local_weights.append(copy.deepcopy(weights))
            local_losses.append(copy.deepcopy(losses))
            model_per_cluster[i] = global_model# = model
            total_data = sum((len(ug) for ugs in user_groups_per_cluster for ug in ugs.values()))
            cluster_data = sum((len(ug) for ug in user_groups_per_cluster[i].values()))
            percentage = cluster_data / total_data
            cluster_percentage.append(percentage)

        # averaging global weights
        global_weights = utils.average_weights(local_weights, cluster_percentage)

        if sigma_global != 0.0:
            dp_xl.apply_noise(global_weights, local_bs, sigma_global, noise, device)

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # ============== EVAL ==============
        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(num_users):
            local_model = update.LocalUpdate(args=args, dataset=train_dataset,
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

        # Test inference after completion of training
        test_acc, test_loss = update.test_inference(args, global_model, test_dataset, dtype=data_type)
        test_accuracies.append(test_acc)
        test_losses.append(test_loss)

    # print(f' \n Results after {epochs} global rounds of training:')
    print(f"\nAvg Training Stats after {epoch} global rounds:")
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    # init the data exporter
    exporter = output.data_exporter(dataset, model, epochs, learning_rate, iid, frac, local_ep, local_bs, num_clusters, appendage=appendage,
                                    sigma_local=sigma_local, sigma_global=sigma_global, sigma_intermediate=sigma_intermediate)

    # Saving the objects train_loss and train_accuracy:
    exporter.dump_file([train_loss, train_accuracy, test_losses, test_accuracies])

    # PLOTTING (optional)
    if plot:
        exporter.plot_all(train_loss, train_accuracy, train=True)
        exporter.plot_all(test_losses, test_accuracies)

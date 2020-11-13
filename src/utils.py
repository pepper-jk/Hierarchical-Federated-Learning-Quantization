#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import numpy as np
import torch
import sys

from torchvision import datasets, transforms

import privacy_engine_xl as dp_xl
import sampling
import update

from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar


def get_dataset(args):
    return _get_dataset(args.dataset, args.iid, args.unequal, args.num_users)

def _get_dataset(dataset, iid, unequal, num_users):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if dataset == 'cifar':
        data_dir = '../data/cifar/'
        dataset_name = "CIFAR10"
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)
    elif dataset == 'mnist':
        data_dir = '../data/mnist/'
        dataset_name = "MNIST"
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)
    else:
        sys.exit("No such dataset: " + dataset)

    # sample training data amongst users
    if iid:
        # Sample IID user data from dataset
        print(f"Dataset: {dataset_name} IID")
        user_groups = sampling.sample_iid(train_dataset, num_users)
    else:
        # Sample Non-IID user data from dataset
        if unequal:
            # Chose unequal splits for every user
            print(f"Dataset: {dataset_name} unequal Non-IID")
            user_groups = sampling.sample_noniid_unequal(train_dataset, num_users)
        else:
            # Chose equal splits for every user
            print(f"Dataset: {dataset_name} equal Non-IID")
            # FIXME: CIFAR10 does not work with non-iid
            user_groups = sampling.sample_noniid(train_dataset, num_users)

    return train_dataset, test_dataset, user_groups


def average_weights(w, data_percentages):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = 0
        for i in range(0, len(w)):
            w_avg[key] += w[i][key] * data_percentages[i] / sum(data_percentages)
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.learning_rate}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return


def set_device(args):
    return _set_device(args.gpu, args.gpu_id)

def _set_device(gpu, gpu_id):
    # Select CPU or GPU
    if not gpu or not torch.cuda.is_available():
        device=torch.device('cpu')
    else:
        # Check that GPU is indeed available
        device = torch.device(gpu_id)

    return device


def build_model(args, train_dataset):
    return _build_model(args.model, args.dataset, args.mlpdim, args.num_classes, args.num_channels, train_dataset)

def _build_model(model, dataset, mlpdim, num_classes, num_channels, train_dataset):
    if model == 'cnn':
        # Convolutional neural network
        if dataset == 'mnist':
            model = CNNMnist(num_channels, num_classes)
        elif dataset == 'fmnist':
            model = CNNFashion_Mnist()
        elif dataset == 'cifar':
            model = CNNCifar(num_classes)

    elif model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
        model = MLP(dim_in=len_in, dim_hidden=mlpdim,
                               dim_out=num_classes)
    else:
        sys.exit('Error- unrecognized model: ' + model)

    return model


def fl_train(args, train_dataset, cluster_global_model, cluster, usergrp, epochs, logger, local_bs, device, sigma=None, noise=None, num_users_per_epoch: int=10, cluster_dtype=torch.float32):
    """
    Defining the training function.
    """

    cluster_train_loss, cluster_train_acc = [], []
    cluster_val_acc_list, cluster_net_list = [], []
    cluster_cv_loss, cluster_cv_acc = [], []
    # print_every = 1
    cluster_val_loss_pre, counter = 0, 0

    for epoch in range(epochs):
        cluster_local_weights, cluster_local_losses = [], []
        local_percentages = []
        # print(f'\n | Cluster Training Round : {epoch+1} |\n')

        cluster_global_model.train()
        m = min(int(len(cluster)), num_users_per_epoch)
        idxs_users = np.random.choice(cluster, m, replace=False)


        for idx in idxs_users:
            cluster_local_model = update.LocalUpdate(args=args, dataset=train_dataset, idxs=usergrp[idx], logger=logger)
            cluster_w, cluster_loss = cluster_local_model.update_weights(model=copy.deepcopy(cluster_global_model), global_round=epoch, dtype=cluster_dtype)
            cluster_local_weights.append(copy.deepcopy(cluster_w))
            cluster_local_losses.append(copy.deepcopy(cluster_loss))
            # print('| Global Round : {} | User : {} | \tLoss: {:.6f}'.format(epoch, idx, cluster_loss))
            user_data = len(usergrp[idx])
            total_data = sum((len(usergrp[idx]) for idx in cluster))
            percentage = user_data / total_data
            local_percentages.append(percentage)

        # averaging global weights
        cluster_global_weights = average_weights(cluster_local_weights, local_percentages)

        if sigma != 0.0:
            dp_xl.apply_noise(cluster_global_weights, local_bs, sigma, noise, device)

        # update global weights
        cluster_global_model.load_state_dict(cluster_global_weights)

        cluster_loss_avg = sum(cluster_local_losses) / len(cluster_local_losses)
        cluster_train_loss.append(cluster_loss_avg)

        # ============== EVAL ==============
        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        cluster_global_model.eval()
        # C = np.random.choice(cluster, m, replace=False) # random set of clients
        # print("C: ", C)
        # for c in C:
        # for c in range(len(cluster)):
        for c in idxs_users:
            cluster_local_model = update.LocalUpdate(args=args, dataset=train_dataset, idxs=usergrp[c], logger=logger)
            # local_model = LocalUpdate(args=args, dataset=train_dataset,idxs=user_groups[idx], logger=logger)
            acc, loss = cluster_local_model.inference(model=cluster_global_model, dtype=cluster_dtype)
            list_acc.append(acc)
            list_loss.append(loss)
        # cluster_train_acc.append(sum(list_acc)/len(list_acc))
        # Add
    # print("Cluster accuracy: ", 100*cluster_train_acc[-1])
    print("Cluster accuracy: ", 100*sum(list_acc)/len(list_acc))

    return cluster_global_model, cluster_global_weights, cluster_loss_avg

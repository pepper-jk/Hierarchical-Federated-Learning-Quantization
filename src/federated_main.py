#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import numpy as np
import os
import pickle
import tensorboardX
import time
import torch
from tqdm import tqdm

import options
import output
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
    device = utils.set_device(args)

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = utils.get_dataset(args)

    # BUILD MODEL
    global_model = utils.build_model(args, train_dataset)

    data_type = torch.float32
    appendage = ''
    # Set model to use Floating Point 16
    if floating_point_16:
        data_type = torch.float16
        appendage = '_FP16'

    # Set the model to train and send it to device.
    global_model.to(device, dtype=data_type)
    global_model.train()
    print(global_model)

    # MODEL PARAM SUMMARY
    pytorch_total_params = sum(p.numel() for p in global_model.parameters())
    print("Model total number of parameters: ", pytorch_total_params)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    test_losses, test_accuracies = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 1
    val_loss_pre, counter = 0, 0
    testacc_check, epoch = 0, 0

    # global training epochs
    for epoch in range(epochs):
        local_weights, local_losses = [], [] # init empty local weights and local losses
        print(f'\n | Global Training Round : {epoch+1} |\n') # starting with | Global Training Round : 1 |

        """
        model.train() tells your model that you are training the model. So effectively layers like dropout, batchnorm etc. which behave different on the train and test procedures know what is going on and hence can behave accordingly.

        More details: It sets the mode to train (see source code). You can call either model.eval() or model.train(mode=False) to tell that you are testing. It is somewhat intuitive to expect train function to train model but it does not do that. It just sets the mode.
        """
        # ============== TRAIN ==============
        global_model.train()
        m = max(int(frac * num_users), 1) # C = frac. Setting number of clients m for training
        idxs_users = np.random.choice(range(num_users), m, replace=False) # num_users=100 total clients. Choosing a random array of indices. Subset of clients.

        for idx in idxs_users: # For each client in the subset.
            local_model = update.LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            w, loss = local_model.update_weights( # update_weights() contain multiple prints
                model=copy.deepcopy(global_model), global_round=epoch, dtype=data_type)
                # w = local model weights
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # Averaging m local client weights
        global_weights = utils.average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg) # Performance measure

        # ============== EVAL ==============
        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval() # must set your model into evaluation mode when computing model output values if dropout or bach norm used for training.

        for c in range(num_users): # 0 to 99
            local_model = update.LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[c], logger=logger)
            acc, loss = local_model.inference(model=global_model, dtype=data_type)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc)) # Performance measure

        # Add
        testacc_check = 100*train_accuracy[-1]
        epoch = epoch + 1

        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0: # If print_every=2, => print every 2 rounds
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

        # Test inference after completion of training
        test_acc, test_loss = update.test_inference(args, global_model, test_dataset, dtype=data_type)
        test_accuracies.append(test_acc)
        test_losses.append(test_loss)

    print(f' \n Results after {epoch} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    # init the data exporter
    exporter = output.data_exporter(dataset, model, epochs, learning_rate, iid, frac, local_ep, local_bs, appendage=appendage)

    # Saving the objects train_loss and train_accuracy:
    exporter.dump_file([train_loss, train_accuracy, test_losses, test_accuracies])

    # PLOTTING (optional)
    if plot:
        exporter.plot_all(train_loss, train_accuracy, train=True)

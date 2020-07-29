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

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details, set_device, build_model


if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = tensorboardX.SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    # Select CPU or GPU
    device = set_device(args)

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # BUILD MODEL
    global_model = build_model(args, train_dataset)

    data_type = torch.float32
    appendage = ''
    # Set model to use Floating Point 16
    if args.floating_point_16:
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
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 1
    val_loss_pre, counter = 0, 0
    testacc_check, epoch = 0, 0

    # global training epochs
    for epoch in range(args.epochs):
        local_weights, local_losses = [], [] # init empty local weights and local losses
        print(f'\n | Global Training Round : {epoch+1} |\n') # starting with | Global Training Round : 1 |

        """
        model.train() tells your model that you are training the model. So effectively layers like dropout, batchnorm etc. which behave different on the train and test procedures know what is going on and hence can behave accordingly.

        More details: It sets the mode to train (see source code). You can call either model.eval() or model.train(mode=False) to tell that you are testing. It is somewhat intuitive to expect train function to train model but it does not do that. It just sets the mode.
        """
        # ============== TRAIN ==============
        global_model.train()
        m = max(int(args.frac * args.num_users), 1) # C = args.frac. Setting number of clients m for training
        idxs_users = np.random.choice(range(args.num_users), m, replace=False) # args.num_users=100 total clients. Choosing a random array of indices. Subset of clients.

        for idx in idxs_users: # For each client in the subset.
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            w, loss = local_model.update_weights( # update_weights() contain multiple prints
                model=copy.deepcopy(global_model), global_round=epoch, dtype=data_type)
                # w = local model weights
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # Averaging m local client weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg) # Performance measure

        # ============== EVAL ==============
        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval() # must set your model into evaluation mode when computing model output values if dropout or bach norm used for training.

        for c in range(args.num_users): # 0 to 99
            local_model = LocalUpdate(args=args, dataset=train_dataset,
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
    test_acc, test_loss = test_inference(args, global_model, test_dataset, dtype=data_type)

    print(f' \n Results after {epoch} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    # Saving the objects train_loss and train_accuracy:
    file_name = '../save/objects{}/FL_{}_{}_{}_lr[{}]_C[{}]_iid[{}]_E[{}]_B[{}]{}.pkl'.\
        format(appendage.lower(), args.dataset, args.model, epoch, args.learning_rate, args.frac,
               args.iid, args.local_ep, args.local_bs, appendage)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    # PLOTTING (optional)
    if args.plot:
        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.use('Agg')

        # Plot Loss curve
        plt.figure()
        plt.title('Training Loss vs Communication rounds')
        plt.plot(range(len(train_loss)), train_loss, color='r')
        plt.ylabel('Training loss')
        plt.xlabel('Communication Rounds')
        plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
                    format(args.dataset, args.model, args.epochs, args.frac,
                        args.iid, args.local_ep, args.local_bs))

        # Plot Average Accuracy vs Communication rounds
        plt.figure()
        plt.title('Average Accuracy vs Communication rounds')
        plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
        plt.ylabel('Average Accuracy')
        plt.xlabel('Communication Rounds')
        plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
                    format(args.dataset, args.model, args.epochs, args.frac,
                        args.iid, args.local_ep, args.local_bs))

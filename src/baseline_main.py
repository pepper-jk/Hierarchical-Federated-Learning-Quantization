#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib.pyplot as plt
import pickle
import sys
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import options
import update
import utils


if __name__ == '__main__':
    args = options.args_parser()
    start_time = time.time()

    # Select CPU or GPU
    device = utils.set_device(args)

    # get cli arguments
    floating_point_16 = args.floating_point_16
    epochs = args.epochs
    optimizer = args.optimizer

    ## for filesave
    model = args.model
    dataset = args.dataset
    learning_rate = args.learning_rate
    iid = args.iid

    ## plots
    plot = args.plot

    # load datasets
    train_dataset, test_dataset, _ = utils.get_dataset(args)

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

    # Training
    # Set optimizer and criterion
    if optimizer == 'sgd':
        optimizer = torch.optim.SGD(global_model.parameters(), lr=learning_rate,
                                    momentum=0.5)
    elif optimizer == 'adam':
        optimizer = torch.optim.Adam(global_model.parameters(), lr=learning_rate,
                                     weight_decay=1e-4)
    elif optimizer == 'adagrad':
        optimizer = torch.optim.Adagrad(global_model.parameters(), lr=learning_rate,
                                     weight_decay=1e-4)
    elif optimizer == 'adamax':
        optimizer = torch.optim.Adamax(global_model.parameters(), lr=learning_rate,
                                     weight_decay=1e-4)
    elif optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(global_model.parameters(), lr=learning_rate,
                                     weight_decay=1e-4)
    else:
        sys.exit('Error- unrecognized optimizer: ' + optimizer)

    trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    criterion = torch.nn.NLLLoss().to(device, dtype=data_type)

    epoch_loss = []

    for epoch in tqdm(range(epochs)):
        batch_loss = []

        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device, dtype=data_type), labels.to(device)

            optimizer.zero_grad()
            outputs = global_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, batch_idx * len(images), len(trainloader.dataset),
                    100. * batch_idx / len(trainloader), loss.item()))
            batch_loss.append(loss.item())

        loss_avg = sum(batch_loss)/len(batch_loss)
        print('\nTrain loss:', loss_avg)
        epoch_loss.append(loss_avg)

    # testing
    test_acc, test_loss = update.test_inference(args, global_model, test_dataset, dtype=data_type)
    print('Test on', len(test_dataset), 'samples')
    print("Test Accuracy: {:.2f}%".format(100*test_acc))

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    # Saving the objects train_loss, test_acc, test_loss:
    file_name = '../save/objects{}/BaseSGD_{}_{}_epoch[{}]_lr[{}]_iid[{}]{}.pkl'.\
        format(appendage.lower(), dataset, model, epoch, learning_rate, iid, appendage)

    with open(file_name, 'wb') as f:
        pickle.dump([epoch_loss, test_acc, test_loss], f)

    # Plot loss
    if plot:
        plt.figure()
        plt.plot(range(len(epoch_loss)), epoch_loss)
        plt.xlabel('epochs')
        plt.ylabel('Train loss')
        plt.savefig('../save/nn_{}_{}_{}.png'.format(dataset, model,
                                                    epochs))

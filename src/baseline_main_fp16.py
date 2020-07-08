#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from utils import get_dataset, set_device, build_model
from options import args_parser
from update import test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
import pickle
import time

from sys import exit
from torchsummary import summary


if __name__ == '__main__':
    args = args_parser()
    start_time = time.time()

    # Select CPU or GPU
    device = set_device(args)

    # load datasets
    train_dataset, test_dataset, _ = get_dataset(args)

    # BUILD MODEL
    global_model = build_model(args, train_dataset)

    # Set the model to train and send it to device.
    global_model.to(device)
    # Set model to use Floating Point 16
    global_model.to(dtype=torch.float16)
    global_model.train()
    print(global_model)
    #img_size = train_dataset[0][0].shape
    #summary(global_model, img_size)  ####
    #print(global_model.parameters())

    # Training
    # Set optimizer and criterion
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr,
                                    momentum=0.5)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(global_model.parameters(), lr=args.lr,
                                     weight_decay=1e-4)
    elif args.optimizer == 'adagrad':
        optimizer = torch.optim.Adagrad(global_model.parameters(), lr=args.lr,
                                     weight_decay=1e-4)
    elif args.optimizer == 'adamax':
        optimizer = torch.optim.Adamax(global_model.parameters(), lr=args.lr,
                                     weight_decay=1e-4)
    elif args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(global_model.parameters(), lr=args.lr,
                                     weight_decay=1e-4)
    else:
        exit('Error- unrecognized optimizer: ' + args.optimizer)

    # look under optim for more info on scheduler
    #scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    criterion = torch.nn.NLLLoss().to(device)
    criterion.to(dtype = torch.float16)

    epoch_loss = []

    for epoch in tqdm(range(args.epochs)):
        batch_loss = []

        for batch_idx, (images, labels) in enumerate(trainloader):
            images=images.to(dtype=torch.float16)
            images, labels = images.to(device), labels.to(device)

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
    test_acc, test_loss = test_inference(args, global_model, test_dataset, dtype=torch.float16)
    print('Test on', len(test_dataset), 'samples')
    print("Test Accuracy: {:.2f}%".format(100*test_acc))


    # Saving the objects train_loss, test_acc, test_loss:
    file_name = '../save/objects_fp16/BaseSGD_{}_{}_epoch[{}]_lr[{}]_iid[{}]_FP16.pkl'.\
        format(args.dataset, args.model, epoch, args.lr, args.iid)

    with open(file_name, 'wb') as f:
        pickle.dump([epoch_loss, test_acc, test_loss], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    # Plot loss
    if args.plot:
        plt.figure()
        plt.plot(range(len(epoch_loss)), epoch_loss)
        plt.xlabel('epochs')
        plt.ylabel('Train loss')
        plt.savefig('../save/nn_{}_{}_{}.png'.format(args.dataset, args.model,
                                                    args.epochs))

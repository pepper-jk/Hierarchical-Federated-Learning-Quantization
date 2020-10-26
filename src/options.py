#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    #parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description="Arguments for Neural Net")

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=5,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=100,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1,
                        help='the fraction of clients per epoch (per cluster): C')
    parser.add_argument('--local_ep', type=int, default=1,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10,
                        help="local batch size: B")
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01,
                        help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5,
                        help="SGD momentum (default: 0.5)")

    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help="model name")
    parser.add_argument('--kernel_num', type=int, default=9,
                        help="number of each kind of kernel")
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help="comma-separated kernel size to \
                        use for convolution")
    parser.add_argument('--num_channels', type=int, default=1, help="number \
                        of channels of imgs")
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                        mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                        strided convolutions")
    parser.add_argument('-fp16', '--floating_point_16', action='store_true', default=False, \
                        help="use floating point 16 for model")

    # differential privacy
    differential_privacy = parser.add_argument_group('differential_privacy',
                                                     'All arguments related to differential privacy.\
                                                      Per default local noise and clipping are used.\
                                                      But if global or intermediate noise is set,\
                                                      local noise is deactivated and needs to be explicitly\
                                                      set in order to coexist.')

    differential_privacy.add_argument('-dp', '--differential_privacy', action='store_true', default=False, \
                        help="use differential-privacy")
    differential_privacy.add_argument('-s', '--sigma', type=float, default=1.0, \
                        help="Noise multiplier (default 1.0)")
    differential_privacy.add_argument('-c', '--clip_max_per_sample_grad_norm', type=float, default=1.0, \
                        help='Clip per-sample gradients to this norm (default 1.0)')
    differential_privacy.add_argument('--alphas', nargs='+', type=float, default=None,
                        help='alpha orders for Renyi Differential Privacy')
    differential_privacy.add_argument('--noise', type=str, default='gaussian', help="type \
                        of noise")

    differential_privacy.add_argument('-sl', '--sigma_local', type=float, default=0.0, \
                        help="Explicit local noise multiplier")
    differential_privacy.add_argument('-si', '--sigma_intermediate', type=float, default=0.0, \
                        help="Explicit intermediate noise multiplier")
    differential_privacy.add_argument('-sg', '--sigma_global', type=float, default=0.0, \
                        help="Explicit global noise multiplier")

    differential_privacy.add_argument('-nl', '--noise_local', action='store_true', default=None,
                                      help="Apply local noise with general sigma value")
    differential_privacy.add_argument('-ni', '--noise_intermediate', action='store_true', default=False,
                                      help="Apply intermediate noise with general sigma value")
    differential_privacy.add_argument('-ng', '--noise_global', action='store_true', default=False,
                                      help="Apply global noise with general sigma value")

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name \
                        of datasetS")
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    parser.add_argument('--gpu', action='store_true', default=False, help="Set to use cuda. \
                        Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")
    parser.add_argument('--iid', type=int, default=1,
                        help="Default set to IID. Set to 0 for non-IID.")
    parser.add_argument('--unequal', action='store_true', default=False,
                        help="whether to use unequal data splits for non-i.i.d setting")
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help="rounds of early stopping")
    parser.add_argument('--verbose', type=int, default=1, help="verbose")
    parser.add_argument('--seed', type=int, default=1, help="random seed")
    parser.add_argument('-p', '--plot', action='store_true', default=False, help="plot the results")

    # Add arguments
    parser.add_argument('--num_clusters', type=int, default=2, help="the number of clusters")
    parser.add_argument('--test_acc', type=int, default=95, help="target test accuracy")
    parser.add_argument('--Cepochs', type=int, default=5,help="number of rounds of training in each cluster")
    parser.add_argument('--mlpdim', type=int, default=200,help="MLP model hidden dimension")
    parser.add_argument('--gpu_id', default='cuda:0', help="To set GPU device \
                        ID if cuda is availlable")
    parser.add_argument('--model_dtype', default='torch.float32', help="Dtype \
                        for model")
    parser.add_argument('--loss_dtype', default='torch.float32', help="Dtype \
                        for loss or criterion")


    args = parser.parse_args()

    if args.unequal == True:
        args.iid = 0

    if args.floating_point_16 and not args.gpu:
        parser.error("floating_point_16 will only work on GPU (CUDA). Please specify --gpu and --gpu_id.")

    # differential privacy

    if args.alphas == None:
        # FIXME: find good default value
        # example: [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
        args.alphas = []

    if args.noise_local or args.noise_global or args.noise_intermediate or args.sigma_local or args.sigma_global or args.sigma_intermediate:
        args.differential_privacy = True

    if args.differential_privacy and args.noise_local == None and args.sigma_local == None:
        if args.noise_intermediate or args.noise_global:
            # the user specified global or intermediate noise without explicitly specifying local noise
            # they most likely do not want local noise
            args.noise_local = False
        else:
            # local noise is the default
            args.noise_local = True

    if args.noise_local and not args.sigma_local:
            args.sigma_local = args.sigma

    if args.noise_intermediate and not args.sigma_intermediate:
            args.sigma_intermediate = args.sigma

    if args.noise_global and not args.sigma_global:
            args.sigma_global = args.sigma

    return args

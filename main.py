# -*- coding: utf_8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import csv
import codecs
import numpy as np
import re
from functools import wraps
from data_loader import train_data_loader, test_data_loader
import math
from DiscCentroidsLoss import create_loss
# Load initial models
from networks import EmbeddingNetwork

# Load batch sampler and train loss
from datasets import BalancedBatchSampler
from losses import BlendedLoss, MAIN_LOSS_CHOICES

from trainer import fit
from inference import retrieve2,batch_process_image,retrieveimage


def load(file_path):
    model.load_state_dict(torch.load(file_path))
    print('model loaded!')
    return model


def infer(model, queries):
    retrieval_results = retrieve2(model, queries, input_size, infer_batch_size)

    return list(zip(range(len(retrieval_results)), retrieval_results.items()))


def get_arguments():
    args = argparse.ArgumentParser()

    args.add_argument('--dataset-path', type=str)
    args.add_argument('--model-save-dir', type=str)
    args.add_argument('--model-to-test', type=str)

    # Hyperparameters
    args.add_argument('--epochs', type=int, default=50)
    args.add_argument('--model', type=str,
                      choices=['densenet161', 'resnet101',  'inceptionv3', 'seresnext'],
                      default='resnet50')
    args.add_argument('--input-size', type=int, default=256, help='size of input image')
    args.add_argument('--num-classes', type=int, default=64, help='number of classes for batch sampler')
    args.add_argument('--num-samples', type=int, default=3, help='number of samples per class for batch sampler')
    args.add_argument('--embedding-dim', type=int, default=1024, help='size of embedding dimension')
    args.add_argument('--feature-extracting', type=bool, default=False)
    args.add_argument('--use-pretrained', type=bool, default=True)
    args.add_argument('--lr', type=float, default=1e-4)
    args.add_argument('--scheduler', type=str, default='MultiStepLR', choices=['StepLR', 'MultiStepLR'])
    args.add_argument('--attention', default=True)
    args.add_argument('--loss-type', type=str, default='n-pair', choices=MAIN_LOSS_CHOICES)
    args.add_argument('--cross-entropy', default=True)
    args.add_argument('--use-augmentation', default=False)

    # Mode selection
    args.add_argument('--mode', type=str, default='test', help='mode selection: train or test.')

    return args.parse_args()


if __name__ == '__main__':
    config = get_arguments()

    dataset_path = config.dataset_path

    # Model parameters
    model_name = config.model
    input_size = config.input_size
    embedding_dim = config.embedding_dim
    feature_extracting = config.feature_extracting
    use_pretrained = config.use_pretrained
    attention_flag = config.attention

    # Training parameters
    nb_epoch = config.epochs
    loss_type = config.loss_type
    cross_entropy_flag = config.cross_entropy
    scheduler_name = config.scheduler
    lr = config.lr

    # Mini-batch parameters
    num_classes = config.num_classes
    num_samples = config.num_samples
    use_augmentation = config.use_augmentation
    infer_batch_size = 64
    log_interval = 200

    """ Model """
    model = EmbeddingNetwork(model_name=model_name,
                             embedding_dim=embedding_dim,
                             feature_extracting=feature_extracting,
                             use_pretrained=use_pretrained,
                             attention_flag=attention_flag,
                             cross_entropy_flag=cross_entropy_flag)
    # model = EmbeddingNetwork(model_name='resnet50',
    #                             embedding_dim=1024,
    #                             feature_extracting=False,
    #                             use_pretrained=True,
    #                             attention_flag=True,
    #                             cross_entropy_flag=True)

    ## pretrained
    # model.load_state_dict(torch.load('./model_fold2/model_acc76'), strict=False)
    with open("./result.txt","a+") as f:
        f.writelines(' embedding_dim:' + str(embedding_dim) \
            + ' loss-type:' + str(loss_type) \
                + ' cross-entropy:' + str(cross_entropy_flag) \
                    + ' use-pretrained:'+str(use_pretrained) \
                        + ' lr:' + str(lr) + '\n')
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)

    model_save_dir = './model_fold/'

    if config.mode == 'train':

        """ Load data """
        # print('dataset path', dataset_path)
        # train_dataset_path = dataset_path + '/train/train_data'
        train_dataset_path = './fold_dataset'
        # train_dataset_path = './train'

        img_dataset = train_data_loader(data_path=train_dataset_path, img_size=input_size,
                                        use_augment=use_augmentation)
        # Balanced batch sampler and online train loader
        train_batch_sampler = BalancedBatchSampler(img_dataset, n_classes=num_classes, n_samples=num_samples)
        online_train_loader = torch.utils.data.DataLoader(img_dataset,
                                                          batch_sampler=train_batch_sampler,
                                                          num_workers=4,
                                                          pin_memory=True)
        device = torch.device("cuda:0")
        # device = torch.device("cpu")
        # Gather the parameters to be optimized/updated.
        # params_to_update = model.parameters()
        print("Params to learn:")
        if feature_extracting:
            params_to_update = []
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if name == 'model_ft.fc.weight': 
                        print(name,param)
                    # params_to_update.append(param)
                    # print("\t", name)
        else:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    print(name)
                    # nn.init.xavier_normal_(param, gain=1)
                    # if name == 'model_ft.fc.weight'or name == 'attention.key_conv.weight'or name == 'attention.query_conv.weight':
                    # if name == 'model_ft.fc.weight':
                    #     # print(name,param)
                    #     nn.init.xavier_normal_(param, gain=1)
                        # print(name,param)
                        
        # Send the model to GPU
        model = model.to(device)

        # Set different learning rates for different parameters
        # ignored_params = list(map(id, model.model_ft.fc.parameters()))
        # base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
        # params_list = [{'params': base_params, 'lr': 1e-4},]
        # params_list.append({'params': model.model_ft.fc.parameters(), 'lr': 0.01})

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        if scheduler_name == 'StepLR':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
        elif scheduler_name == 'MultiStepLR':
            if use_augmentation:
                scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 30], gamma=0.1)
            else:
                scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15, 20], gamma=0.1)
        else:
            raise ValueError('Invalid scheduler')

        # Loss function
        loss_fn = BlendedLoss(loss_type, cross_entropy_flag)

        # Train (fine-tune) model
        fit(online_train_loader, model, loss_fn, optimizer, scheduler, nb_epoch, cross_entropy_flag,
            device=device, log_interval=log_interval, save_model_to=model_save_dir,train_dataset=img_dataset)

    elif config.mode == 'test':
        test_dataset_path = './test_LEsample/'
        # test_dataset_path = './test/'
        # test_files = [os.path.join(test_dataset_path, path) for path in os.listdir(test_dataset_path)]
        test_files = [os.path.join(test_dataset_path, path) for path in os.listdir(test_dataset_path)]
        # test_files2 = []
        # for path in test_files:
        #     test_files2.extend([os.path.join(path, path2) for path2 in os.listdir(path)])
        model.load_state_dict(torch.load('./model_fold2/model_acc77'), strict=False)
        retrieve2(model, test_files, True, 256, 64)
        

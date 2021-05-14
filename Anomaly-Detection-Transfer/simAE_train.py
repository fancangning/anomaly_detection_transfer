#!/usr/bin/python
# ****************************************************************#
# ScriptName: simAE_train.py
# Author: fancangning.fcn@alibaba-inc.com
# Create Date: 2021-03-09 11:49
# Modify Author: fancangning.fcn@alibaba-inc.com
# Modify Date: 2021-03-09 11:49
# Function: knowledge transfer by memory item similarity
# ***************************************************************#

import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.nn.init as init
import torch.utils.data as data
import torch.utils.data.dataset as dataset
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.utils as v_utils
import torchvision.datasets
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
import cv2
import math
from collections import OrderedDict
import copy
import time
import pandas as pd
import seaborn as sns

import data.utils as data_utils
import models.loss as loss
from models import AutoEncoderCov2D, AutoEncoderCov2DMem, AutoEncoderCov2DMemFace, AdversarialAutoEncoderCov2D

import argparse

parser = argparse.ArgumentParser(description="MemoryNormality")
parser.add_argument('--gpu', type=str, default='0', help='the gpu to use')
parser.add_argument('--batch_size', type=int, default=12, help='batch size for training')
parser.add_argument('--epochs', type=int, default=80, help='number of epochs for training')
parser.add_argument('--val_epoch', type=int, default=2, help='evaluate the model every %d epoch')
parser.add_argument('--h', type=int, default=28, help='height of input images')
parser.add_argument('--w', type=int, default=28, help='width of input images')
parser.add_argument('--c', type=int, default=1, help='channel of input images')
parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
parser.add_argument('--ModelName', help='AE/MemAE/MemAEFace/AdversarialAE', type=str, default='MemAE')
parser.add_argument('--ModelSetting', help='Conv2D/Conv2DSpar', type=str,
                    default='Conv2DSpar')  # give the layer details later
parser.add_argument('--MemDim', help='Memory Dimention', type=int, default=100)
parser.add_argument('--EntropyLossWeight', help='EntropyLossWeight', type=float, default=0.0002)
parser.add_argument('--ShrinkThres', help='ShrinkThres', type=float, default=0.025)
parser.add_argument('--num_workers', type=int, default=4, help='number of workers for the train loader')
parser.add_argument('--num_workers_test', type=int, default=1, help='number of workers for the test loader')
parser.add_argument('--source_dataset', type=str, default='USPS', help='source dataset')
parser.add_argument('--target_dataset', type=str, default='MNIST', help='target dataset')
parser.add_argument('--dataset_path', type=str, default='../data/', help='directory of data')
parser.add_argument('--exp_dir', type=str, default='./simAE_log/', help='directory of log')
parser.add_argument('--version', type=int, default=0, help='experiment version')
parser.add_argument('--anomaly_rate', type=float, default=0.25)

args = parser.parse_args()

torch.manual_seed(2020)

torch.backends.cudnn.enabled = True  # make sure to use cudnn for computational performance
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare image transform
if args.source_dataset == 'celeb10w' or args.source_dataset == 'crop_SVHN':
    s_img_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # [0, 1] -> [-1, 1]
    ])
elif args.source_dataset == 'SVHN':
    s_img_transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # [0, 1] -> [-1, 1]
    ])
elif args.source_dataset == 'MNIST':
    s_img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # [0, 1] -> [-1, 1]
    ])
elif args.source_dataset == 'USPS':
    s_img_transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # [0, 1] -> [-1, 1]
    ])
else:
    raise Exception('invalid source dataset')

if args.target_dataset == 'celeb10w' or args.target_dataset == 'crop_SVHN':
    t_img_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # [0, 1] -> [-1, 1]
    ])
elif args.target_dataset == 'SVHN':
    t_img_transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # [0, 1] -> [-1, 1]
    ])
elif args.target_dataset == 'MNIST':
    t_img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # [0, 1] -> [-1, 1]
    ])
elif args.target_dataset == 'USPS':
    t_img_transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # [0, 1] -> [-1, 1]
    ])
else:
    raise Exception('invalid target dataset')

# prepare dataset
if args.source_dataset == 'MNIST':
    s_dataset = torchvision.datasets.MNIST(args.dataset_path, download=True, train=True, transform=s_img_transform)
    s_train, s_val = data.random_split(s_dataset, [50000, 10000], generator=torch.Generator().manual_seed(2020))
elif args.source_dataset == 'celeb10w':
    s_dataset = data_utils.Celeb10wDataset(args.dataset_path, transform=s_img_transform)
    s_train, s_val, s_test = data.random_split(s_dataset, [50000, 10000, 11114], generator=torch.Generator().manual_seed(2020))
elif args.source_dataset == 'SVHN':
    s_dataset = torchvision.datasets.SVHN(args.dataset_path, split='train', transform=s_img_transform, download=True)
    s_train, s_val = data.random_split(s_dataset, [60000, 13257], generator=torch.Generator().manual_seed(2020))
elif args.source_dataset == 'crop_SVHN':
    s_dataset = torchvision.datasets.ImageFolder(os.path.join(args.dataset_path, 'train_crop'), transform=s_img_transform)
    s_train, s_val = data.random_split(s_dataset, [30000, 4274], generator=torch.Generator().manual_seed(2020))
elif args.source_dataset == 'USPS':
    s_dataset = torchvision.datasets.USPS(args.dataset_path, train=True, transform=s_img_transform, download=True)
    s_train, s_val = data.random_split(s_dataset, [7000, 291], generator=torch.Generator().manual_seed(2020))
else:
    raise Exception('invalid source dataset')

s_train = data_utils.AnomalyDataset(s_train, [0], 0)
s_val = data_utils.AnomalyDataset(s_val, [0], 0)

if args.target_dataset == 'MNIST':
    t_dataset = torchvision.datasets.MNIST(args.dataset_path, download=True, train=True, transform=t_img_transform)
    t_train, t_val = data.random_split(t_dataset, [50000, 10000], generator=torch.Generator().manual_seed(2020))
elif args.target_dataset == 'celeb10w':
    t_dataset = data_utils.Celeb10wDataset(args.dataset_path, transform=t_img_transform)
    t_train, t_val, t_test = data.random_split(t_dataset, [50000, 10000, 11114],
                                               generator=torch.Generator().manual_seed(2020))
elif args.target_dataset == 'SVHN':
    t_dataset = torchvision.datasets.SVHN(args.dataset_path, split='train', transform=t_img_transform, download=True)
    t_train, t_val = data.random_split(t_dataset, [60000, 13257], generator=torch.Generator().manual_seed(2020))
elif args.target_dataset == 'crop_SVHN':
    t_dataset = torchvision.datasets.ImageFolder(os.path.join(args.dataset_path, 'train_crop'),
                                                 transform=t_img_transform)
    t_train, t_val = data.random_split(t_dataset, [30000, 4274], generator=torch.Generator().manual_seed(2020))
elif args.target_dataset == 'USPS':
    t_dataset = torchvision.datasets.USPS(args.dataset_path, train=True, transform=t_img_transform, download=True)
    t_train, t_val = data.random_split(t_dataset, [7000, 291], generator=torch.Generator().manual_seed(2020))
else:
    raise Exception('valid target dataset')

t_train = data_utils.AnomalyDataset(t_train, [0], args.anomaly_rate)
t_val = data_utils.AnomalyDataset(t_val, [0], args.anomaly_rate)

print('The source domain', args.source_dataset)
print('The target domain', args.target_dataset)

s_train_batch = data.DataLoader(s_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                drop_last=True)
s_val_batch = data.DataLoader(s_val, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              drop_last=True)

t_train_batch = data.DataLoader(t_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                drop_last=True)
t_val_batch = data.DataLoader(t_val, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              drop_last=True)

s_train_batch = list(s_train_batch)
s_val_batch = list(s_val_batch)
t_train_batch = list(t_train_batch)
t_val_batch = list(t_val_batch)

print('The number of source batch', len(s_train_batch))
print('The number of source validation batch', len(s_val_batch))
print('The number of target batch', len(t_train_batch))
print('The number of target validation batch', len(t_val_batch))

log_dir = os.path.join(args.exp_dir, args.source_dataset + 'to' + args.target_dataset + '_' + args.ModelName,
                       'lr_%.5f_entropyloss_%.5f_anomalyrate_%.5f_version_%d' % (
                           args.lr, args.EntropyLossWeight, args.anomaly_rate, args.version))
writer = SummaryWriter(
    os.path.join('simAE_runs', args.source_dataset + 'to' + args.target_dataset + '_' + args.ModelName,
                 'lr_%.5f_entropyloss_%.5f_anomalyrate_%.5f_version_%d' % (
                     args.lr, args.EntropyLossWeight, args.anomaly_rate, args.version)))

if not os.path.exists(log_dir):
    os.makedirs(log_dir)
orig_stdout = sys.stdout
f = open(os.path.join(log_dir, 'log.txt'), 'w')
sys.stdout = f

for arg in vars(args):
    print(arg, getattr(args, arg))

if args.ModelName == 'MemAE':
    model = AutoEncoderCov2DMem(args.c, args.MemDim, shrink_thres=args.ShrinkThres)
else:
    raise Exception('Wrong model name')

model = model.to(device)
parameter_list = [p for p in model.parameters() if p.requires_grad]

for name, p in model.named_parameters():
    if not p.requires_grad:
        print('---------NO GRADIENT-----', name)

optimizer = torch.optim.Adam(parameter_list, lr=args.lr)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40], gamma=0.2)

print('memory size:')
print(model.mem_rep.memory.weight.size())
print('memory:')
print(model.mem_rep.memory.weight)
print(model.mem_rep.memory.weight.data)
print('----------------------------------------------------------------------------')

# Train phase 1

for epoch in range(args.epochs):
    print('epoch/total epoch:' + str(epoch) + '/' + str(args.epochs))
    model.train()
    tr_re_loss, tr_mem_loss, tr_tot = 0.0, 0.0, 0.0

    for data in s_train_batch:
        img = data[0].to(device)
        optimizer.zero_grad()

        model_output = model(img)
        recons, attr = model_output['output'], model_output['att']
        re_loss, re_loss_per_sample = loss.get_reconstruction_loss(img, recons, mean=0.5, std=0.5)
        mem_loss = loss.get_memory_loss(attr)
        tot_loss = re_loss + mem_loss * args.EntropyLossWeight
        tr_re_loss += re_loss.data.item()
        tr_mem_loss += mem_loss.data.item()
        tr_tot += tot_loss.data.item()

        tot_loss.backward()
        optimizer.step()

    scheduler.step()

    # Save the model
    if epoch >= args.epochs - 50:
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            torch.save(model.state_dict(), log_dir + '/model-{:04d}.pt'.format(epoch))

# Train phase 2

memory_item = model.mem_rep.memory.weight.data
print('memory_item size:', memory_item.size())
memory_item_mean = torch.mean(memory_item, 0, True)
print('memory_item_mean size:', memory_item_mean.size())
memory_item_mean = memory_item_mean.expand(args.batch_size, -1)
print('memory_item_mean_expand:', memory_item_mean.size())
print('-----------------------------------------------------------------')

similarity_total = list()
target_anomaly_label_total = list()
for data in t_train_batch:
    img = data[0].to(device)
    model_output = model(img)
    recons, att = model_output['output'], model_output['att']
    att = att.reshape(args.batch_size, 3, 3, args.MemDim)

    sim = torch.mean(att, [1, 2, 3])
    similarity_total.append(torch.squeeze(sim))
    target_anomaly_label_total.append(torch.squeeze(data[1]))

similarity_total = torch.cat(similarity_total, 0)
target_anomaly_label_total = torch.cat(target_anomaly_label_total, 0)

similarity_total = similarity_total.detach().cpu()
target_anomaly_label_total = target_anomaly_label_total.detach().cpu()
print('similarity_total size:', similarity_total.size())
print('target_anomaly_label_total size:', target_anomaly_label_total.size())
# print('similarity_total:', similarity_total)
# print('target_anomaly_label_total:', target_anomaly_label_total)

target_normal_mask = target_anomaly_label_total.eq(0)
target_abnormal_mask = target_anomaly_label_total.eq(1)
print('target_normal_mask size:', target_normal_mask.size())
print('target_abnormal_mask size:', target_abnormal_mask.size())

target_normal_similarity = torch.masked_select(similarity_total, target_normal_mask)
target_abnormal_similarity = torch.masked_select(similarity_total, target_abnormal_mask)
print('target_normal_similarity size:', target_normal_similarity.size())
print('target_abnormal_similarity size:', target_abnormal_similarity.size())

max_length = max(len(target_normal_similarity), len(target_abnormal_similarity))
if len(target_normal_similarity) == max_length:
    target_abnormal_similarity = torch.cat(
        [target_abnormal_similarity.cpu(), torch.tensor([np.nan] * (max_length - len(target_abnormal_similarity)))])
elif len(target_abnormal_similarity) == max_length:
    target_normal_similarity = torch.cat(
        [target_normal_similarity.cpu(), torch.tensor([np.nan] * (max_length - len(target_normal_similarity)))])
else:
    raise Exception('Wrong length')

df = pd.DataFrame(
    {
        'target_normal_similarity': target_normal_similarity,
        'target_abnormal_similarity': target_abnormal_similarity,
    }
)
fig, ax = plt.subplots()
sns.kdeplot(data=df, fill=True, common_norm=False, palette='crest', alpha=.5, linewidth=0)
fig.savefig(os.path.join(log_dir, 'similarity_kde.png'))

#!/usr/bin/python
#****************************************************************#
# ScriptName: deAE_train.py
# Author: fancangning.fcn@alibaba-inc.com
# Create Date: 2021-03-16 20:18
# Modify Author: fancangning.fcn@alibaba-inc.com
# Modify Date: 2021-03-16 20:18
# Function: train double encoder autoencoder
#***************************************************************#

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
from models import AutoEncoderCov2D, AutoEncoderCov2DMem, AutoEncoderCov2DMemFace, AdversarialAutoEncoderCov2D, DoubleEncoderAutoEncoderCov2D

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
parser.add_argument('--ModelName', help='AE/MemAE/MemAEFace/AdversarialAE/DoubleEncoderAE', type=str, default='DoubleEncoderAE')
parser.add_argument('--AdversarialLossWeight', help='AdversarialLossWeight', type=float, default=0.01)
parser.add_argument('--num_workers', type=int, default=4, help='number of workers for the train loader')
parser.add_argument('--source_dataset', type=str, default='USPS', help='source dataset')
parser.add_argument('--target_dataset', type=str, default='MNIST', help='target dataset')
parser.add_argument('--dataset_path', type=str, default='../data/', help='directory of data')
parser.add_argument('--exp_dir', type=str, default='./deAE_log/', help='directory of log')
parser.add_argument('--version', type=int, default=0, help='experiment version')
parser.add_argument('--anomaly_rate', type=float, default=0.25)
parser.add_argument('--u', type=float, default=1, help='parameters that control the size of the reversal gradient')
parser.add_argument('--alpha', type=float, default=1, help='parameters that control the size of the reversal gradient')

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
    raise Exception('valid source dataset')

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
    raise Exception('valid target dataset')

# prepare dataset
if args.source_dataset == 'MNIST':
    s_dataset = torchvision.datasets.MNIST(args.dataset_path, download=True, train=True, transform=s_img_transform)
    s_train, s_val = data.random_split(s_dataset, [50000, 10000], generator=torch.Generator().manual_seed(2020))
elif args.source_dataset == 'celeb10w':
    s_dataset = data_utils.Celeb10wDataset(args.dataset_path, transform=s_img_transform)
    s_train, s_val, s_test = data.random_split(s_dataset, [50000, 10000, 11114],
                                               generator=torch.Generator().manual_seed(2020))
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
    raise Exception('valid source dataset')

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

t_train1 = data_utils.AnomalyDataset(t_train, [0], 0)
t_train = data_utils.AnomalyDataset(t_train, [0], args.anomaly_rate)
t_val = data_utils.AnomalyDataset(t_val, [0], args.anomaly_rate)

print('The source domain', args.source_dataset)
print('The target domain', args.target_dataset)

s_train_batch = data.DataLoader(s_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                drop_last=True)
s_val_batch = data.DataLoader(s_val, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              drop_last=True)
t_train1_batch = data.DataLoader(t_train1, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                 drop_last=True)
t_train_batch = data.DataLoader(t_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                drop_last=True)
t_val_batch = data.DataLoader(t_val, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              drop_last=True)

s_train_batch = list(s_train_batch)
s_val_batch = list(s_val_batch)
t_train1_batch = list(t_train1_batch)
t_train_batch = list(t_train_batch)
t_val_batch = list(t_val_batch)

print('The number of source batch', len(s_train_batch))
print('The number of source validation batch', len(s_val_batch))
print('The number of target batch', len(t_train_batch))
print('The number of target validation batch', len(t_val_batch))

log_dir = os.path.join(args.exp_dir, args.source_dataset + 'to' + args.target_dataset + '_' + args.ModelName,
                           'lr_%.5f_adversariallossweight_%.5f_anomalyrate_%.5f_version_%d' % (
                               args.lr, args.AdversarialLossWeight, args.anomaly_rate, args.version))
writer = SummaryWriter(os.path.join('deAE_runs', args.source_dataset + 'to' + args.target_dataset + '_' + args.ModelName,
                                    'lr_%.5f_adversariallossweight_%.5f_anomalyrate_%.5f_version_%d' % (
                                        args.lr, args.AdversarialLossWeight, args.anomaly_rate, args.version)))

if not os.path.exists(log_dir):
    os.makedirs(log_dir)
orig_stdout = sys.stdout
file = open(os.path.join(log_dir, 'log.txt'), 'w')
sys.stdout = file

for arg in vars(args):
    print(arg, getattr(args, arg))

if args.ModelName == 'DoubleEncoderAE':
    model = DoubleEncoderAutoEncoderCov2D(args.c, 0.0)
else:
    raise Exception('Wrong model name')

model = model.to(device)
parameter_list = [p for p in model.parameters() if p.requires_grad]

for name, p in model.named_parameters():
    if not p.requires_grad:
        print('---------NO GRADIENT-----', name)

# pretrain
pretrain_optimizer = torch.optim.Adam([
    {'params': model.encoder.parameters()},
    {'params': model.decoder.parameters()},
    {'params': model.encoder2.parameters()},
], lr=args.lr)
pretrain_scheduler = optim.lr_scheduler.MultiStepLR(pretrain_optimizer, milestones=[40], gamma=0.2)

for epoch in range(args.epochs):
    print('epoch/total epoch:' + str(epoch) + '/' + str(args.epochs))
    model.train()
    tr_img_re_loss, tr_embedding_re_loss, tr_adversarial_loss, tr_tot = 0.0, 0.0, 0.0, 0.0

    for idx in range(max(len(s_train_batch), len(t_train_batch), len(t_train1_batch))):

        s_idx = idx % len(s_train_batch)
        t_idx = idx % len(t_train_batch)
        t1_idx = idx % len(t_train1_batch)

        s_data = s_train_batch[s_idx]
        t_data = t_train_batch[t_idx]
        t1_data = t_train1_batch[t1_idx]

        img = torch.cat((s_data[0], t_data[0]), 0)
        img = img.to(device)

        model_output = model(img)

        recons, y0, y0_orig, f, f2 = model_output['recons'], model_output['y0'], model_output['y0_orig'], model_output['f'], model_output['f2']
        img_re_loss, img_re_loss_per_sample = loss.get_reconstruction_loss(img, recons, mean=0.5, std=0.5)
        embedding_re_loss, embedding_re_loss_per_sample = loss.get_reconstruction_loss(f[:args.batch_size], f2[:args.batch_size], mean=0.5, std=0.5)

        tot_loss = img_re_loss + embedding_re_loss

        pretrain_optimizer.zero_grad()
        tot_loss.backward()
        pretrain_optimizer.step()
    pretrain_scheduler.step()

# forward and visualize the distribution of recon loss(pretrain)
target_anomaly_label_total = list()
target_recon_loss_total = dict()
source_recon_loss_total = dict()
target_recon_loss_total['img'] = list()
target_recon_loss_total['embedding'] = list()
source_recon_loss_total['img'] = list()
source_recon_loss_total['embedding'] = list()

model.eval()
for idx in range(len(s_train_batch)):
    s_data = s_train_batch[idx]

    img = s_data[0]
    img = img.to(device)

    model_output = model(img)

    recons, f, f2 = model_output['recons'], model_output['f'], model_output['f2']
    img_re_loss, img_re_loss_per_sample = loss.get_reconstruction_loss(img, recons, mean=0.5, std=0.5)
    embedding_re_loss, embedding_re_loss_per_sample = loss.get_reconstruction_loss(f, f2, mean=0.5, std=0.5)

    source_recon_loss_total['img'].append(torch.squeeze(img_re_loss_per_sample))
    source_recon_loss_total['embedding'].append(torch.squeeze(embedding_re_loss_per_sample))

for idx in range(len(t_train_batch)):
    t_data = t_train_batch[idx]

    img = t_data[0]
    img = img.to(device)

    model_output = model(img)

    recons, f, f2 = model_output['recons'], model_output['f'], model_output['f2']
    img_re_loss, img_re_loss_per_sample = loss.get_reconstruction_loss(img, recons, mean=0.5, std=0.5)
    embedding_re_loss, embedding_re_loss_per_sample = loss.get_reconstruction_loss(f, f2, mean=0.5, std=0.5)

    target_recon_loss_total['img'].append(torch.squeeze(img_re_loss_per_sample))
    target_recon_loss_total['embedding'].append(torch.squeeze(embedding_re_loss_per_sample))

    target_anomaly_label_total.append(torch.squeeze(t_data[1]))

target_anomaly_label_total = torch.cat(target_anomaly_label_total, 0).detach().cpu()
target_recon_loss_total['img'] = torch.cat(target_recon_loss_total['img'], 0).detach().cpu()
target_recon_loss_total['embedding'] = torch.cat(target_recon_loss_total['embedding'], 0).detach().cpu()
source_recon_loss_total['img'] = torch.cat(source_recon_loss_total['img'], 0).detach().cpu()
source_recon_loss_total['embedding'] = torch.cat(source_recon_loss_total['embedding'], 0).detach().cpu()

target_normal_mask = target_anomaly_label_total.eq(0)
target_abnormal_mask = target_anomaly_label_total.eq(1)

# visualize the distribution of img recon loss
target_normal_img_recon = torch.masked_select(target_recon_loss_total['img'], target_normal_mask)
target_abnormal_img_recon = torch.masked_select(target_recon_loss_total['img'], target_abnormal_mask)
target_normal_embedding_recon = torch.masked_select(target_recon_loss_total['embedding'], target_normal_mask)
target_abnormal_embedding_recon = torch.masked_select(target_recon_loss_total['embedding'], target_abnormal_mask)

max_length = max(len(target_normal_img_recon), len(target_abnormal_img_recon), len(source_recon_loss_total['img']))
print()
if len(target_normal_img_recon) == max_length:
    source_recon_loss_total['img'] = torch.cat([source_recon_loss_total['img'].cpu(),
                                                torch.tensor([np.nan] * (max_length - len(source_recon_loss_total['img'])))])
    target_abnormal_img_recon = torch.cat([target_abnormal_img_recon.cpu(), torch.tensor([np.nan] * (max_length - len(target_abnormal_img_recon)))])
elif len(target_abnormal_img_recon) == max_length:
    source_recon_loss_total['img'] = torch.cat([source_recon_loss_total['img'].cpu(),
                                                torch.tensor([np.nan] * (max_length - len(source_recon_loss_total['img'])))])
    target_normal_img_recon = torch.cat([target_normal_img_recon.cpu(), torch.tensor([np.nan] * (max_length - len(target_normal_img_recon)))])
elif len(source_recon_loss_total['img']) == max_length:
    target_normal_img_recon = torch.cat([target_normal_img_recon.cpu(), torch.tensor([np.nan] * (max_length - len(target_normal_img_recon)))])
    target_abnormal_img_recon = torch.cat([target_abnormal_img_recon.cpu(), torch.tensor([np.nan] * (max_length - len(target_abnormal_img_recon)))])
else:
    raise Exception('wrong length')
df = pd.DataFrame(
    {
        'target_normal_img_recon': target_normal_img_recon,
        'target_abnormal_img_recon': target_abnormal_img_recon,
        'source_img_recon': source_recon_loss_total['img'],
    }
)
fig, ax = plt.subplots()
sns.kdeplot(data=df, fill=True, common_norm=False, palette='crest', alpha=.5, linewidth=0)
fig.savefig(os.path.join(log_dir, 'pretrain_img_recon_kde.png'))
writer.add_figure('pretrain_img_recon_kde/AdversarialLossWeight_' + str(args.AdversarialLossWeight), fig)

# visualize the distribution of embedding recon loss
target_normal_embedding_recon = torch.masked_select(target_recon_loss_total['embedding'], target_normal_mask)
target_abnormal_embedding_recon = torch.masked_select(target_recon_loss_total['embedding'], target_abnormal_mask)

max_length = max(len(target_normal_embedding_recon), len(target_abnormal_embedding_recon), len(source_recon_loss_total['embedding']))
if len(target_normal_embedding_recon) == max_length:
    source_recon_loss_total['embedding'] = torch.cat([source_recon_loss_total['embedding'].cpu(),
                                                torch.tensor([np.nan] * (max_length - len(source_recon_loss_total['embedding'])))])
    target_abnormal_embedding_recon = torch.cat([target_abnormal_embedding_recon.cpu(),
                                                 torch.tensor([np.nan] * (max_length - len(target_abnormal_embedding_recon)))])
elif len(target_abnormal_embedding_recon) == max_length:
    source_recon_loss_total['embedding'] = torch.cat([source_recon_loss_total['embedding'].cpu(),
                                                torch.tensor([np.nan] * (max_length - len(source_recon_loss_total['embedding'])))])
    target_normal_embedding_recon = torch.cat([target_normal_embedding_recon.cpu(),
                                               torch.tensor([np.nan] * (max_length - len(target_normal_embedding_recon)))])
elif len(source_recon_loss_total['embedding']) == max_length:
    target_normal_embedding_recon = torch.cat([target_normal_embedding_recon.cpu(),
                                               torch.tensor([np.nan] * (max_length - len(target_normal_embedding_recon)))])
    target_abnormal_embedding_recon = torch.cat([target_abnormal_embedding_recon.cpu(),
                                                 torch.tensor([np.nan] * (max_length - len(target_abnormal_embedding_recon)))])
else:
    raise Exception('wrong length')
df = pd.DataFrame(
    {
        'target_normal_embedding_recon': target_normal_embedding_recon,
        'target_abnormal_embedding_recon': target_abnormal_embedding_recon,
        'source_embedding_recon': source_recon_loss_total['embedding'],
    }
)
fig, ax = plt.subplots()
sns.kdeplot(data=df, fill=True, common_norm=False, palette='crest', alpha=.5, linewidth=0)
fig.savefig(os.path.join(log_dir, 'pretrain_embedding_recon_kde.png'))
writer.add_figure('pretrain_embedding_recon_kde/AdversarialLossWeight_' + str(args.AdversarialLossWeight), fig)

optimizer = torch.optim.Adam([
    {'params': model.encoder.parameters()},
    {'params': model.decoder.parameters()},
    {'params': model.encoder2.parameters()},
    {'params': model.adnet0.parameters()}
], lr=args.lr)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40], gamma=0.2)

for epoch in range(args.epochs):
    print('epoch/total epoch:' + str(epoch) + '/' + str(args.epochs))
    model.train()
    tr_img_re_loss, tr_embedding_re_loss, tr_ad_loss0_s, tr_ad_loss0_t, tr_ad_loss0, tr_tot = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    target_anomaly_label_total = list()
    target_recon_loss_total = dict()
    target_recon_loss_total['img'] = list()
    target_recon_loss_total['embedding'] = list()
    source_recon_loss_total = dict()
    source_recon_loss_total['img'] = list()
    source_recon_loss_total['embedding'] = list()
    y0_orig_total = dict()
    y0_orig_total['source'] = list()
    y0_orig_total['target'] = list()

    for idx in range(max(len(s_train_batch), len(t_train_batch), len(t_train1_batch))):
        iter_num = epoch * max(len(s_train_batch), len(t_train_batch), len(t_train1_batch)) + idx
        total_iter_num = args.epochs * max(len(s_train_batch), len(t_train_batch), len(t_train1_batch))
        model.adnet0.coeff = np.float(2.0 * args.u / (1.0 + np.exp(-args.alpha * iter_num / total_iter_num)) - args.u)
        writer.add_scalar('model.adnet0.coeff/AdversarialLossWeight_' + str(args.AdversarialLossWeight),
                          model.adnet0.coeff, iter_num)

        s_idx = idx % len(s_train_batch)
        t_idx = idx % len(t_train_batch)
        t1_idx = idx % len(t_train1_batch)

        s_data = s_train_batch[s_idx]
        t_data = t_train_batch[t_idx]
        t1_data = t_train1_batch[t1_idx]


        img = torch.cat((s_data[0], t1_data[0]), 0)
        a_img = torch.cat((s_data[0], t_data[0]), 0)
        domain_label = torch.tensor([0] * len(s_data[0]) + [1] * len(t_data[0]), dtype=torch.float).view(-1, 1)

        img = img.to(device)
        a_img = a_img.to(device)
        domain_label = domain_label.to(device)

        model_output = model(img)
        recons, y0, y0_orig, f, f2 = model_output['recons'], model_output['y0'], model_output['y0_orig'], model_output['f'], model_output['f2']

        a_model_output = model(a_img)
        a_recons, a_f, a_f2 = a_model_output['recons'], a_model_output['f'], a_model_output['f2']

        # calculate loss
        img_re_loss, _ = loss.get_reconstruction_loss(img, recons, is_weighted_recon=True, mean=0.5, std=0.5)
        _, img_re_loss_per_sample = loss.get_reconstruction_loss(a_img, a_recons, mean=0.5, std=0.5)
        embedding_re_loss, _ = loss.get_reconstruction_loss(f, f2, mean=0.5, std=0.5)
        _, embedding_re_loss_per_sample = loss.get_reconstruction_loss(a_f, a_f2, mean=0.5, std=0.5)

        target_anomaly_label_total.append(torch.squeeze(t_data[1]))
        target_recon_loss_total['img'].append(torch.squeeze(img_re_loss_per_sample)[args.batch_size:])
        target_recon_loss_total['embedding'].append(torch.squeeze(embedding_re_loss_per_sample)[args.batch_size:])
        source_recon_loss_total['img'].append(torch.squeeze(img_re_loss_per_sample)[:args.batch_size])
        source_recon_loss_total['embedding'].append(torch.squeeze(embedding_re_loss_per_sample)[:args.batch_size])
        y0_orig_total['source'].append(torch.squeeze(y0_orig)[:args.batch_size])
        y0_orig_total['target'].append(torch.squeeze(y0_orig)[args.batch_size:])

        # calculate weight
        t_recons = img_re_loss_per_sample[args.batch_size:]
        weight = (t_recons - torch.min(t_recons)) / torch.max(t_recons)
        weight = torch.where(weight > 1e-2, weight, torch.tensor(1e-2).float().to(device))
        weight = -1.0 * torch.log(weight).detach()
        weight = (weight - torch.min(weight)) / torch.max(weight)
        weight = torch.where(weight > 1e-8, weight, torch.tensor(0.).float().to(device))
        weight = weight.unsqueeze(1)
        weight = weight.detach()
        print('adversarial weight:', weight)
        print('BCELoss_t:', nn.BCELoss(reduction='none')(y0[len(s_data[0]):], domain_label[len(s_data[0]):]))
        print('y0:', y0)
        print('domain_label:', domain_label)
        print('-------------------------------------------------------------------------------------------------')

        ad_loss0_s = nn.BCELoss()(y0[:len(s_data[0])], domain_label[:len(s_data[0])])
        ad_loss0_t = torch.mean(weight * nn.BCELoss(reduction='none')(y0[len(s_data[0]):], domain_label[len(s_data[0]):]))
        ad_loss0 = ad_loss0_s + ad_loss0_t
        tot_loss = img_re_loss + embedding_re_loss + args.AdversarialLossWeight * ad_loss0

        tr_img_re_loss += img_re_loss.data.item()
        tr_embedding_re_loss += embedding_re_loss.data.item()
        tr_ad_loss0_s += ad_loss0_s.data.item()
        tr_ad_loss0_t += ad_loss0_t.data.item()
        tr_ad_loss0 += ad_loss0.data.item()
        tr_tot += tot_loss.data.item()

        optimizer.zero_grad()
        tot_loss.backward()
        optimizer.step()
    scheduler.step()
    writer.add_scalar('tr_img_re_loss/AdversarialLossWeight_' + str(args.AdversarialLossWeight),
                      tr_img_re_loss/max(len(s_train_batch), len(t_train_batch)), epoch)
    writer.add_scalar('tr_embedding_re_loss/AdversarialLossWeight_' + str(args.AdversarialLossWeight),
                      tr_embedding_re_loss/max(len(s_train_batch), len(t_train_batch)), epoch)
    writer.add_scalar('tr_ad_loss0_s/AdversarialLossWeight_' + str(args.AdversarialLossWeight),
                      tr_ad_loss0_s/max(len(s_train_batch), len(t_train_batch)), epoch)
    writer.add_scalar('tr_ad_loss0_t/AdversarialLossWeight_' + str(args.AdversarialLossWeight),
                      tr_ad_loss0_t/max(len(s_train_batch), len(t_train_batch)), epoch)
    writer.add_scalar('tr_ad_loss0/AdversarialLossWeight_' + str(args.AdversarialLossWeight),
                      tr_ad_loss0/max(len(s_train_batch), len(t_train_batch)), epoch)
    writer.add_scalar('tr_tot/AdversarialLossWeight_' + str(args.AdversarialLossWeight),
                      tr_tot/max(len(s_train_batch), len(t_train_batch)), epoch)
    if epoch == 0 or epoch % 10 == 9:
        target_anomaly_label_total = torch.cat(target_anomaly_label_total, 0)
        target_recon_loss_total['img'] = torch.cat(target_recon_loss_total['img'], 0)
        target_recon_loss_total['embedding'] = torch.cat(target_recon_loss_total['embedding'], 0)
        source_recon_loss_total['img'] = torch.cat(source_recon_loss_total['img'], 0)
        source_recon_loss_total['embedding'] = torch.cat(source_recon_loss_total['embedding'], 0)
        y0_orig_total['source'] = torch.cat(y0_orig_total['source'], 0)
        y0_orig_total['target'] = torch.cat(y0_orig_total['target'], 0)

        target_anomaly_label_total = target_anomaly_label_total.detach().cpu()
        target_recon_loss_total['img'] = target_recon_loss_total['img'].detach().cpu()
        target_recon_loss_total['embedding'] = target_recon_loss_total['embedding'].detach().cpu()
        source_recon_loss_total['img'] = source_recon_loss_total['img'].detach().cpu()
        source_recon_loss_total['embedding'] = source_recon_loss_total['embedding'].detach().cpu()
        y0_orig_total['source'] = y0_orig_total['source'].detach().cpu()
        y0_orig_total['target'] = y0_orig_total['target'].detach().cpu()

        target_normal_mask = target_anomaly_label_total.eq(0)
        target_abnormal_mask = target_anomaly_label_total.eq(1)

        target_normal_img_recon = torch.masked_select(target_recon_loss_total['img'], target_normal_mask)
        target_abnormal_img_recon = torch.masked_select(target_recon_loss_total['img'], target_abnormal_mask)
        target_normal_embedding_recon = torch.masked_select(target_recon_loss_total['embedding'], target_normal_mask)
        target_abnormal_embedding_recon = torch.masked_select(target_recon_loss_total['embedding'], target_abnormal_mask)
        target_normal_y0_orig = torch.masked_select(y0_orig_total['target'], target_normal_mask)
        target_abnormal_y0_orig = torch.masked_select(y0_orig_total['target'], target_abnormal_mask)

        # img recon distribution
        max_length = max(len(target_normal_img_recon), len(target_abnormal_img_recon), len(source_recon_loss_total['img']))
        if len(target_normal_img_recon) == max_length:
            source_recon_loss_total['img'] = torch.cat(
                [source_recon_loss_total['img'].cpu(), torch.tensor([np.nan] * (max_length - len(source_recon_loss_total['img'])))])
            target_abnormal_img_recon = torch.cat(
                [target_abnormal_img_recon.cpu(), torch.tensor([np.nan] * (max_length - len(target_abnormal_img_recon)))])
        elif len(target_abnormal_img_recon) == max_length:
            source_recon_loss_total['img'] = torch.cat(
                [source_recon_loss_total['img'].cpu(), torch.tensor([np.nan] * (max_length - len(source_recon_loss_total['img'])))])
            target_normal_img_recon = torch.cat(
                [target_normal_img_recon.cpu(), torch.tensor([np.nan] * (max_length - len(target_normal_img_recon)))])
        elif len(source_recon_loss_total['img']) == max_length:
            target_normal_img_recon = torch.cat(
                [target_normal_img_recon.cpu(), torch.tensor([np.nan] * (max_length - len(target_normal_img_recon)))])
            target_abnormal_img_recon = torch.cat(
                [target_abnormal_img_recon.cpu(), torch.tensor([np.nan] * (max_length - len(target_abnormal_img_recon)))])
        else:
            raise Exception('wrong length')
        df = pd.DataFrame(
            {
                'target_normal_img_recon': target_normal_img_recon,
                'target_abnormal_img_recon': target_abnormal_img_recon,
                'source_img_recon': source_recon_loss_total['img']
            }
        )
        fig, ax = plt.subplots()
        sns.kdeplot(data=df, fill=True, common_norm=False, palette='crest', alpha=.5, linewidth=0)
        fig.savefig(os.path.join(log_dir, 'img_recon_kde.png'))
        writer.add_figure('img_recon_kde/AdversarialLossWeight_' + str(args.AdversarialLossWeight) + '/epoch_' + str(epoch), fig)

        # embedding recon distribution
        max_length = max(len(target_normal_embedding_recon), len(target_abnormal_embedding_recon),
                         len(source_recon_loss_total['embedding']))
        if len(target_normal_embedding_recon) == max_length:
            source_recon_loss_total['embedding'] = torch.cat(
                [source_recon_loss_total['embedding'].cpu(), torch.tensor([np.nan] * (max_length - len(source_recon_loss_total['embedding'])))])
            target_abnormal_embedding_recon = torch.cat(
                [target_abnormal_embedding_recon.cpu(), torch.tensor([np.nan] * (max_length - len(target_abnormal_embedding_recon)))])
        elif len(target_abnormal_embedding_recon) == max_length:
            source_recon_loss_total['embedding'] = torch.cat(
                [source_recon_loss_total['embedding'].cpu(), torch.tensor([np.nan] * (max_length - len(source_recon_loss_total['embedding'])))])
            target_normal_embedding_recon = torch.cat(
                [target_normal_embedding_recon.cpu(), torch.tensor([np.nan] * (max_length - len(target_normal_embedding_recon)))])
        elif len(source_recon_loss_total['embedding']) == max_length:
            target_normal_embedding_recon = torch.cat(
                [target_normal_embedding_recon.cpu(), torch.tensor([np.nan] * (max_length - len(target_normal_embedding_recon)))])
            target_abnormal_embedding_recon = torch.cat(
                [target_abnormal_embedding_recon.cpu(), torch.tensor([np.nan] * (max_length - len(target_abnormal_embedding_recon)))])
        else:
            raise Exception('wrong length')
        df = pd.DataFrame(
            {
                'target_normal_embedding_recon': target_normal_embedding_recon,
                'target_abnormal_embedding_recon': target_abnormal_embedding_recon,
                'source_embedding_recon': source_recon_loss_total['embedding']
            }
        )
        fig, ax = plt.subplots()
        sns.kdeplot(data=df, fill=True, common_norm=False, palette='crest', alpha=.5, linewidth=0)
        fig.savefig(os.path.join(log_dir, 'embedding_recon_kde.png'))
        writer.add_figure(
            'embedding_recon_kde/AdversarialLossWeight_' + str(args.AdversarialLossWeight) + '/epoch_' + str(epoch), fig)

        # domain discriminator0 distribution
        max_length = max(len(target_normal_y0_orig), len(target_abnormal_y0_orig), len(y0_orig_total['source']))
        if len(target_normal_y0_orig) == max_length:
            y0_orig_total['source'] = torch.cat(
                [y0_orig_total['source'].cpu(), torch.tensor([np.nan] * (max_length - len(y0_orig_total['source'])))])
            target_abnormal_y0_orig = torch.cat(
                [target_abnormal_y0_orig.cpu(), torch.tensor([np.nan] * (max_length - len(target_abnormal_y0_orig)))])
        elif len(target_abnormal_y0_orig) == max_length:
            y0_orig_total['source'] = torch.cat(
                [y0_orig_total['source'].cpu(), torch.tensor([np.nan] * (max_length - len(y0_orig_total['source'])))])
            target_normal_y0_orig = torch.cat(
                [target_normal_y0_orig.cpu(), torch.tensor([np.nan] * (max_length - len(target_normal_y0_orig)))])
        elif len(y0_orig_total['source']) == max_length:
            target_normal_y0_orig = torch.cat(
                [target_normal_y0_orig.cpu(), torch.tensor([np.nan] * (max_length - len(target_normal_y0_orig)))])
            target_abnormal_y0_orig = torch.cat(
                [target_abnormal_y0_orig.cpu(), torch.tensor([np.nan] * (max_length - len(target_abnormal_y0_orig)))])
        else:
            raise Exception('wrong length')
        df = pd.DataFrame(
            {
                'target_normal_y0_orig': target_normal_y0_orig,
                'target_abnormal_y0_orig': target_abnormal_y0_orig,
                'source_y0_orig': y0_orig_total['source']
            }
        )
        fig, ax = plt.subplots()
        sns.kdeplot(data=df, fill=True, common_norm=False, palette='crest', alpha=.5, linewidth=0)
        fig.savefig(os.path.join(log_dir, 'y0_kde.png'))
        writer.add_figure('y0_kde/AdversarialLossWeight_' + str(args.AdversarialLossWeight) + '/epoch_' + str(epoch), fig)
    # Save the model
    if epoch >= args.epochs - 50:
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            torch.save(model.state_dict(), log_dir + '/model-{:04d}.pt'.format(epoch))

sys.stdout = orig_stdout
writer.close()
file.close()

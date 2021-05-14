#!/usr/bin/python
#****************************************************************#
# ScriptName: pretrain.py
# Author: fancangning.fcn@alibaba-inc.com
# Create Date: 2021-03-22 19:49
# Modify Author: fancangning@alibaba-inc.com
# Modify Date: 2021-03-22 19:49
# Function: explore a pretrain method to grarantee the fixed distributions of recon loss
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
from models import AutoEncoderCov2D, AutoEncoderCov2DMem, AutoEncoderCov2DMemFace, AdversarialAutoEncoderCov2D

import argparse


def exp_lr_scheduler(optimizer, step, init_lr=1e-2, lr_decay_step=1, step_decay_weight=0.95):

    # Decay learning rate by a factor of step_decay_weight every lr_decay_step
    current_lr = init_lr * (step_decay_weight ** (step / lr_decay_step))

    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr

    return optimizer

parser = argparse.ArgumentParser(description="MemoryNormality")
parser.add_argument('--gpu', type=str, default='0', help='the gpu to use')
parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs for training')
parser.add_argument('--truncated_epochs', type=int, default=40, help='number of truncated epochs')
parser.add_argument('--val_epoch', type=int, default=50, help='evaluate the model every %d epoch')
parser.add_argument('--h', type=int, default=28, help='height of input images')
parser.add_argument('--w', type=int, default=28, help='width of input images')
parser.add_argument('--c', type=int, default=3, help='channel of input images')
parser.add_argument('--lr', type=float, default=1e-2, help='initial learning rate')
parser.add_argument('--step_decay_weight', type=float, default=0.95, help='step_decay_weight')
parser.add_argument('--lr_decay_step', type=int, default=1, help='lr_decay_step')
parser.add_argument('--ModelName', help='AE/MemAE/MemAEFace/AdversarialAE', type=str, default='AdversarialAE')
parser.add_argument('--AdversarialLossWeight', help='AdversarialLossWeight', type=float, default=5e-4)
parser.add_argument('--num_workers', type=int, default=4, help='number of workers for the train loader')
parser.add_argument('--num_workers_test', type=int, default=1, help='number of workers for the test loader')
parser.add_argument('--source_dataset', type=str, default='MNIST', help='source dataset')
parser.add_argument('--target_dataset', type=str, default='USPS', help='target dataset')
parser.add_argument('--dataset_path', type=str, default='../data/', help='directory of data')
parser.add_argument('--exp_dir', type=str, default='./pretrain_log/', help='directory of log')
parser.add_argument('--version', type=int, default=0, help='experiment version')
parser.add_argument('--fine_tune', type=bool, default=False, help='whether fine_tune or not')
parser.add_argument('--fine_tune_model', type=str,
                    default='/home/fancangning/project/Anomaly-Detection-Transfer/log/USPStoMNIST/lr_0.00020_entropyloss_0.00020_version_0/model-0079.pt',
                    help='the base model to fine tune')
parser.add_argument('--anomaly_rate', type=float, default=0.25)
parser.add_argument('--trade_off', type=float, default=1.0)
parser.add_argument('--normal_classes', type=int, default=0)

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
        data_utils.ToThreeChannel(),
        transforms.Normalize((0.5,), (0.5,))  # [0, 1] -> [-1, 1]
    ])
elif args.source_dataset == 'USPS':
    s_img_transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        data_utils.ToThreeChannel(),
        transforms.Normalize((0.5,), (0.5,))  # [0, 1] -> [-1, 1]
    ])
elif args.source_dataset == 'MNIST_M':
    s_img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
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
        data_utils.ToThreeChannel(),
        transforms.Normalize((0.5,), (0.5,))  # [0, 1] -> [-1, 1]
    ])
elif args.target_dataset == 'USPS':
    t_img_transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        data_utils.ToThreeChannel(),
        transforms.Normalize((0.5,), (0.5,))  # [0, 1] -> [-1, 1]
    ])
elif args.target_dataset == 'MNIST_M':
    t_img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
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
elif args.source_dataset == 'MNIST_M':
    s_dataset = data_utils.MNISTM(args.dataset_path, download=True, train=True, transform=s_img_transform)
    s_train, s_val = data.random_split(s_dataset, [50000, 10000], generator=torch.Generator().manual_seed(2020))
else:
    raise Exception('valid source dataset')
if args.fine_tune:
    s_train = data_utils.AnomalyDataset(s_train, [args.normal_classes], args.anomaly_rate)
    s_val = data_utils.AnomalyDataset(s_val, [args.normal_classes], args.anomaly_rate)
else:
    s_train = data_utils.AnomalyDataset(s_train, [args.normal_classes], 0)
    s_val = data_utils.AnomalyDataset(s_val, [args.normal_classes], 0)

if args.target_dataset == 'MNIST':
    t_dataset = torchvision.datasets.MNIST(args.dataset_path, download=True, train=True, transform=t_img_transform)
    t_train, t_val = data.random_split(t_dataset, [50000, 10000], generator=torch.Generator().manual_seed(2020))
elif args.target_dataset == 'celeb10w':
    t_dataset = data_utils.Celeb10wDataset(args.dataset_path, transform=t_img_transform)
    t_train, t_val, t_test = data.random_split(t_dataset, [50000, 10000, 11114], generator=torch.Generator().manual_seed(2020))
elif args.target_dataset == 'SVHN':
    t_dataset = torchvision.datasets.SVHN(args.dataset_path, split='train', transform=t_img_transform, download=True)
    t_train, t_val = data.random_split(t_dataset, [60000, 13257], generator=torch.Generator().manual_seed(2020))
elif args.target_dataset == 'crop_SVHN':
    t_dataset = torchvision.datasets.ImageFolder(os.path.join(args.dataset_path, 'train_crop'), transform=t_img_transform)
    t_train, t_val = data.random_split(t_dataset, [30000, 4274], generator=torch.Generator().manual_seed(2020))
elif args.target_dataset == 'USPS':
    t_dataset = torchvision.datasets.USPS(args.dataset_path, train=True, transform=t_img_transform, download=True)
    t_train, t_val = data.random_split(t_dataset, [7000, 291], generator=torch.Generator().manual_seed(2020))
elif args.target_dataset == 'MNIST_M':
    t_dataset = data_utils.MNISTM(args.dataset_path, train=True, transform=t_img_transform, download=True)
    t_train, t_val = data.random_split(t_dataset, [50000, 10000], generator=torch.Generator().manual_seed(2020))
else:
    raise Exception('valid target dataset')
if args.fine_tune:
    t_train = data_utils.AnomalyDataset(t_train, [args.normal_classes], args.anomaly_rate)
    t_val = data_utils.AnomalyDataset(t_val, [args.normal_classes], args.anomaly_rate)
else:
    t_train = data_utils.AnomalyDataset(t_train, [args.normal_classes], args.anomaly_rate)
    t_val = data_utils.AnomalyDataset(t_val, [args.normal_classes], args.anomaly_rate)

print('The source domain', args.source_dataset)
print('The target domain', args.target_dataset)

s_train_batch = data.DataLoader(s_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
s_val_batch = data.DataLoader(s_val, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
t_train_batch = data.DataLoader(t_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
t_val_batch = data.DataLoader(t_val, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

s_train_batch = list(s_train_batch)
s_val_batch = list(s_val_batch)
t_train_batch = list(t_train_batch)
t_val_batch = list(t_val_batch)

print('The number of source batch', len(s_train_batch))
print('The number of source validation batch', len(s_val_batch))
print('The number of target batch', len(t_train_batch))
print('The number of target validation batch', len(t_val_batch))

# Report the training process
if args.fine_tune:
    log_dir = os.path.join(args.exp_dir,
                           args.source_dataset + 'to' + args.target_dataset + '_' + args.ModelName + '_finetune',
                           'lr_%.5f_trade_off_%.5f_anomalyrate_%.5f_version_%d' % (
                               args.lr, args.trade_off, args.anomaly_rate, args.version))
    writer = SummaryWriter(
        os.path.join('pretrain_runs', args.source_dataset + 'to' + args.target_dataset + '_' + args.ModelName + '_finetune',
                     'lr_%.5f_trade_off_%.5f_anomalyrate_%.5f_version_%d' % (
                         args.lr, args.trade_off, args.anomaly_rate, args.version)))
else:
    log_dir = os.path.join(args.exp_dir, args.source_dataset + 'to' + args.target_dataset + '_' + args.ModelName,
                           'lr_%.5f_trade_off_%.5f_anomalyrate_%.5f_version_%d' % (
                               args.lr, args.trade_off, args.anomaly_rate, args.version))
    writer = SummaryWriter(os.path.join('pretrain_runs', args.source_dataset + 'to' + args.target_dataset + '_' + args.ModelName,
                                        'lr_%.5f_trade_off_%.5f_anomalyrate_%.5f_version_%d' % (
                                            args.lr, args.trade_off, args.anomaly_rate, args.version)))

if not os.path.exists(log_dir):
    os.makedirs(log_dir)
orig_stdout = sys.stdout
f = open(os.path.join(log_dir, 'log.txt'), 'w')
sys.stdout = f

for arg in vars(args):
    print(arg, getattr(args, arg))

# Model setting

if args.ModelName == 'AE':
    model = AutoEncoderCov2D(args.c)
elif args.ModelName == 'AdversarialAE':
    model = AdversarialAutoEncoderCov2D(args.c, backward_coeff=0.0)
elif args.ModelName == 'MemAE':
    model = AutoEncoderCov2DMem(args.c, args.MemDim, shrink_thres=args.ShrinkThres)
elif args.ModelName == 'MemAEFace':
    model = AutoEncoderCov2DMemFace(args.c, args.MemDim, shrink_thres=args.ShrinkThres)
else:
    raise Exception('Wrong model name')

if args.fine_tune:
    model_para = torch.load(os.path.join(log_dir, args.fine_tune_model))
    model.load_state_dict(model_para)

model = model.to(device)

# Training
if args.ModelName == 'AdversarialAE':
    # pretrain AE by source data
    pretrain_optimizer = torch.optim.Adam([
        {'params': model.encoder.parameters()},
        {'params': model.decoder.parameters()}
    ], lr=args.lr)

    for epoch in tqdm(range(args.epochs)):
        print('pretrain epoch/total epoch:' + str(epoch) + '/' + str(args.epochs))
        model.train()
        pretrain_optimizer = exp_lr_scheduler(pretrain_optimizer, epoch, init_lr=args.lr,
                                              lr_decay_step=args.lr_decay_step,
                                              step_decay_weight=args.step_decay_weight)

        for idx in range(max(len(s_train_batch), len(t_train_batch))):

            s_idx = idx % len(s_train_batch)
            t_idx = idx % len(t_train_batch)

            s_data = s_train_batch[s_idx]
            t_data = t_train_batch[t_idx]

            img = torch.cat((s_data[0], t_data[0]), 0)
            img = img.to(device)

            model_output = model(img)

            recons = model_output['out']
            s_re_loss, s_re_loss_per_sample = loss.get_reconstruction_loss(img[:args.batch_size], recons[:args.batch_size], mean=0.5, std=0.5)
            t_re_loss, t_re_loss_per_sample = loss.get_reconstruction_loss(img[args.batch_size:], recons[args.batch_size:], mean=0.5, std=0.5)
            tot_loss = s_re_loss + args.trade_off * t_re_loss
            pretrain_optimizer.zero_grad()
            tot_loss.backward()
            pretrain_optimizer.step()
            if idx == max(len(s_train_batch), len(t_train_batch)) - 1:
                writer.add_scalar('s_re_loss/trade_off_' + str(args.trade_off), s_re_loss, epoch)
                writer.add_scalar('t_re_loss/trade_off_' + str(args.trade_off), t_re_loss, epoch)

    torch.save(model.state_dict(), log_dir + '/model-pretrain.pt')
    target_anomaly_label_total = list()
    target_recon_loss_total = list()
    source_recon_loss_total = list()

    model.eval()
    for idx in range(len(s_train_batch)):
        s_data = s_train_batch[idx]

        img = s_data[0]
        img = img.to(device)

        model_output = model(img)

        recons = model_output['out']

        re_loss, re_loss_per_sample = loss.get_reconstruction_loss(img, recons, mean=0.5, std=0.5)

        source_recon_loss_total.append(torch.squeeze(re_loss_per_sample))

    for idx in range(len(t_train_batch)):
        t_data = t_train_batch[idx]

        img = t_data[0]
        img = img.to(device)

        model_output = model(img)

        recons = model_output['out']
        re_loss, re_loss_per_sample = loss.get_reconstruction_loss(img, recons, mean=0.5, std=0.5)
        target_anomaly_label_total.append(torch.squeeze(t_data[1]))
        target_recon_loss_total.append(torch.squeeze(re_loss_per_sample))

    target_anomaly_label_total = torch.cat(target_anomaly_label_total, 0)
    target_recon_loss_total = torch.cat(target_recon_loss_total, 0)
    source_recon_loss_total = torch.cat(source_recon_loss_total, 0)

    target_anomaly_label_total = target_anomaly_label_total.detach().cpu()
    target_recon_loss_total = target_recon_loss_total.detach().cpu()
    source_recon_loss_total = source_recon_loss_total.detach().cpu()

    target_normal_mask = target_anomaly_label_total.eq(0)
    target_abnormal_mask = target_anomaly_label_total.eq(1)

    target_normal_recon = torch.masked_select(target_recon_loss_total, target_normal_mask)
    target_abnormal_recon = torch.masked_select(target_recon_loss_total, target_abnormal_mask)

    max_length = max(len(target_normal_recon), len(target_abnormal_recon), len(source_recon_loss_total))
    if len(target_normal_recon) == max_length:
        source_recon_loss_total = torch.cat(
            [source_recon_loss_total.cpu(), torch.tensor([np.nan] * (max_length - len(source_recon_loss_total)))])
        target_abnormal_recon = torch.cat(
            [target_abnormal_recon.cpu(), torch.tensor([np.nan] * (max_length - len(target_abnormal_recon)))])
    elif len(target_abnormal_recon) == max_length:
        source_recon_loss_total = torch.cat(
            [source_recon_loss_total.cpu(), torch.tensor([np.nan] * (max_length - len(source_recon_loss_total)))])
        target_normal_recon = torch.cat(
            [target_normal_recon.cpu(), torch.tensor([np.nan] * (max_length - len(target_normal_recon)))])
    elif len(source_recon_loss_total) == max_length:
        target_normal_recon = torch.cat(
            [target_normal_recon.cpu(), torch.tensor([np.nan] * (max_length - len(target_normal_recon)))])
        target_abnormal_recon = torch.cat(
            [target_abnormal_recon.cpu(), torch.tensor([np.nan] * (max_length - len(target_abnormal_recon)))])
    else:
        raise Exception('wrong length')
    df = pd.DataFrame(
        {
            'target_normal_recon': target_normal_recon,
            'target_abnormal_recon': target_abnormal_recon,
            'source_recon': source_recon_loss_total,
        }
    )
    fig, ax = plt.subplots()
    sns.kdeplot(data=df, fill=True, common_norm=False, palette='crest', alpha=.5, linewidth=0)
    fig.savefig(os.path.join(log_dir, 'pretrain_recon_kde.png'))
    writer.add_figure('pretrain_recon_kde/trade_off_' + str(args.trade_off), fig)

    # visualize images/recons and embedding
    embedding_tot_list = list()
    label_tot_list = list()
    img_tot_list = list()
    for idx in range(max(len(s_train_batch), len(t_train_batch))):
        s_idx, t_idx = idx % len(s_train_batch), idx % len(t_train_batch)
        s_data, t_data = s_train_batch[s_idx], t_train_batch[t_idx]
        s_img, t_img = s_data[0], t_data[0]
        s_img, t_img = s_img.to(device), t_img.to(device)

        s_model_output, t_model_output = model(s_img), model(t_img)
        s_recons, t_recons = s_model_output['out'], t_model_output['out']
        s_embedding, t_embedding = s_model_output['f'], t_model_output['f']

        # unormalized
        s_img, t_img = s_img.mul(0.5).add(0.5), t_img.mul(0.5).add(0.5)
        s_recons, t_recons = s_recons.mul(0.5).add(0.5), t_recons.mul(0.5).add(0.5)
        print('s_img:')
        print(s_img[0])
        print('s_recons:')
        print(s_recons[0])
        print('t_img:')
        print(t_img[0])
        print('t_recons:')
        print(t_recons[0])
        print('---------------------------------------------------------------------------------------------------')

        embedding_tot_list.append(s_embedding)
        embedding_tot_list.append(t_embedding)
        label_tot_list += [0] * len(s_img)
        label_tot_list += [1] * len(t_img)
        img_tot_list.append(s_img)
        img_tot_list.append(t_img)

        if idx == 0:
            img_tot = torch.cat([s_img, s_recons, t_img, t_recons], 0)
            grid_img = torchvision.utils.make_grid(img_tot, nrow=args.batch_size)
            torchvision.utils.save_image(grid_img, os.path.join(log_dir, 'img_recon.png'))
            writer.add_image('img_recon/trade_off_' + str(args.trade_off), grid_img)
        if idx == 9:
            break
    embedding_tot_list = torch.cat(embedding_tot_list, 0)
    label_tot_list = label_tot_list
    img_tot_list = torch.cat(img_tot_list, 0)

    embedding_tot_list = embedding_tot_list.detach().cpu()
    img_tot_list = img_tot_list.detach().cpu()
    writer.add_embedding(embedding_tot_list, metadata=label_tot_list, label_img=img_tot_list,
                         tag='embedding/trade_off_' + str(args.trade_off))
else:
    raise Exception('invaild Model Name')

sys.stdout = orig_stdout
writer.close()
f.close()

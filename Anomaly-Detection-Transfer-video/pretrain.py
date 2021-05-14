#!/usr/bin/python
# ****************************************************************#
# ScriptName: pretrain.py
# Author: fancangning.fcn@alibaba-inc.com
# Create Date: 2021-03-27 16:32
# Modify Author: fancangning.fcn@alibaba-inc.com
# Modify Date: 2021-03-27 16:32
# Function: Train adversarial AE for video data
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
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as v_utils
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
from torch.utils.tensorboard import SummaryWriter
import cv2
import math
from collections import OrderedDict
import copy
import time
import pandas as pd
import seaborn as sns

import data.utils as data_utils
import models.loss as loss
import utils
from models import AutoEncoderCov3D, AutoEncoderCov3DMem, AdversarialAutoEncoderCov3DMem

import argparse


def exp_lr_scheduler(optimizer, step, init_lr=1e-2, lr_decay_step=1, step_decay_weight=0.95):
    # Decay learning rate by a factor of step_decay_weight every lr_decay_step
    current_lr = init_lr * (step_decay_weight ** (step / lr_decay_step))

    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr

    return optimizer


parser = argparse.ArgumentParser(description="MemoryNormality")
parser.add_argument('--gpu', type=str, default='0', help='which gpu to use')
parser.add_argument('--batch_size', type=int, default=8, help='batch size for training')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs for training')
parser.add_argument('--val_epoch', type=int, default=50, help='evaluate the model every %d epoch')
parser.add_argument('--h', type=int, default=256, help='height of input images')
parser.add_argument('--w', type=int, default=256, help='width of input images')
parser.add_argument('--c', type=int, default=1, help='channel of input images')
parser.add_argument('--lr', type=float, default=5e-3, help='initial learning rate')
parser.add_argument('--step_decay_weight', type=float, default=0.95, help='step_decay_weight')
parser.add_argument('--lr_decay_step', type=int, default=1, help='lr_decay_step')
parser.add_argument('--t_length', type=int, default=16, help='length of the frame sequences')
parser.add_argument('--ModelName', help='AE/MemAE/AdversarialAE', type=str, default='AdversarialAE')
parser.add_argument('--AdversarialLossWeight', help='AdversarialLossWeight', type=float, default=5e-4)
parser.add_argument('--MemDim', help='Memory Dimention', type=int, default=2000)
parser.add_argument('--EntropyLossWeight', help='EntropyLossWeight', type=float, default=0.0002)
parser.add_argument('--trade_off', type=float, default=0.0)
parser.add_argument('--ShrinkThres', help='ShrinkThres', type=float, default=0.0025)
parser.add_argument('--Suffix', help='Suffix', type=str, default='Non')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers for the train loader')
parser.add_argument('--num_workers_test', type=int, default=1, help='number of workers for the test loader')
parser.add_argument('--source_dataset', type=str, default='UCSDped1', help='type of dataset: UCSDped1, UCSDped2')
parser.add_argument('--target_dataset', type=str, default='UCSDped2', help='type of dataset: UCSDped2, UCSDped1')
parser.add_argument('--source_test_label_path', type=str, default='./ckpt/UCSDped1_gt.npy')
parser.add_argument('--target_test_label_path', type=str, default='./ckpt/UCSDped2_gt.npy')
parser.add_argument('--dataset_path', type=str, default='./data/', help='directory of data')
parser.add_argument('--exp_dir', type=str, default='log', help='directory of log')
parser.add_argument('--version', type=int, default=0, help='experiment version')
parser.add_argument('--use_pretrain_model', type=bool, default=True, help='whether use pretrained model or not')
parser.add_argument('--pretrain_model', type=str, default='model-pretrain.pt')
parser.add_argument('--u', type=float, default=1, help='parameters that control the size of the reversal gradient')
parser.add_argument('--alpha', type=float, default=1, help='parameters that control the size of the reversal gradient')
parser.add_argument('--lambda_value', type=float, default=-1e6, help='parameters that control the shape of sigmoid(used for weighting)')
parser.add_argument('--beta_low', type=float, default=325, help='parameters that control the position of sigmoid(used for weighting)')
parser.add_argument('--beta_high', type=float, default=330, help='parameters that control the position of sigmoid(used for weighting')

args = parser.parse_args()

torch.manual_seed(2020)

torch.backends.cudnn.enabled = True  # make sure to use cudnn for computational performance
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# the path of each dataset
s_train, s_test = data_utils.give_data_folder(args.source_dataset, args.dataset_path)
t_train, t_test = data_utils.give_data_folder(args.target_dataset, args.dataset_path)

# prepare image transform
s_frame_trans = data_utils.give_frame_trans(args.source_dataset, [args.h, args.w])
t_frame_trans = data_utils.give_frame_trans(args.target_dataset, [args.h, args.w])

# prepare dataset
# s_test_label = np.load(args.source_test_label_path, allow_pickle=True)
t_test_label = np.load(args.target_test_label_path, allow_pickle=True)

s_train_dataset = data_utils.DataLoader(s_train, s_frame_trans, None, True, time_step=args.t_length - 1, num_pred=1,
                                        video_start=1, video_end=5)
# s_test_dataset = data_utils.DataLoader(s_test, s_frame_trans, s_test_label, False, time_step=args.t_length - 1, num_pred=1)

t_train_dataset = data_utils.DataLoader(t_train, t_frame_trans, None, True, time_step=args.t_length - 1, num_pred=1,
                                        video_start=1, video_end=4)
t_test_dataset = data_utils.DataLoader(t_test, t_frame_trans, t_test_label, False, time_step=args.t_length - 1,
                                       num_pred=1, video_start=1, video_end=2)

t_train_dataset = torch.utils.data.ConcatDataset([t_train_dataset, t_test_dataset])
# prepare dataloader
s_train_batch = data.DataLoader(s_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                drop_last=True)
# s_test_batch = data.DataLoader(s_test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

t_train_batch = data.DataLoader(t_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                drop_last=True)
# t_test_batch = data.DataLoader(t_test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

s_train_batch = list(s_train_batch)
# s_test_batch = list(s_test_batch)
t_train_batch = list(t_train_batch)
# t_test_batch = list(t_test_batch)
print('the number of source batch:', len(s_train_batch))
print('the number of target batch:', len(t_train_batch))

# report the training process
log_dir = os.path.join(args.exp_dir, args.source_dataset + 'to' + args.target_dataset + '_' + args.ModelName,
                       'lr_%.5f_AdversarialLossWeight_%.5f_version_%d' %
                       (args.lr, args.AdversarialLossWeight, args.version))
writer = SummaryWriter(
    os.path.join('pretrain_runs', args.source_dataset + 'to' + args.target_dataset + '_' + args.ModelName,
                 'lr_%.5f_AdversarialLossWeight_%.5f_version_%d' %
                 (args.lr, args.AdversarialLossWeight, args.version)))
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

orig_stdout = sys.stdout
f = open(os.path.join(log_dir, 'log.txt'), 'w')
sys.stdout = f

for arg in vars(args):
    print(arg, getattr(args, arg))

# model setting

if args.ModelName == 'AdversarialAE':
    model = AdversarialAutoEncoderCov3DMem(args.c, backward_coeff=0.0, mem_dim=args.MemDim,
                                           shrink_thres=args.ShrinkThres)
else:
    raise Exception('Wrong model name')

model = model.to(device)

# Pretrain
if args.ModelName == 'AdversarialAE':
    pretrain_optimizer = torch.optim.Adam([
        {'params': model.encoder.parameters()},
        {'params': model.decoder.parameters()},
        {'params': model.mem_rep.parameters()},
        {'params': model.adaptiveAvgPool.parameters()}
    ], lr=args.lr)

    if args.use_pretrain_model:
        model_para = torch.load(os.path.join(log_dir, args.pretrain_model))
        model.load_state_dict(model_para)
    else:
        for epoch in range(args.epochs):
            print('pretrain epoch/total epoch:' + str(epoch) + '/' + str(args.epochs))
            model.train()
            pretrain_optimizer = exp_lr_scheduler(pretrain_optimizer, epoch, init_lr=args.lr,
                                                  lr_decay_step=args.lr_decay_step,
                                                  step_decay_weight=args.step_decay_weight)
            for idx in range(max(len(s_train_batch), len(t_train_batch))):
                # print('pretrain iter/total iter:' + str(idx) + '/' + str(max(len(s_train_batch), len(t_train_batch))))

                s_idx = idx % len(s_train_batch)
                t_idx = idx % len(t_train_batch)

                s_data = s_train_batch[s_idx]
                t_data = t_train_batch[t_idx]

                img = torch.cat((s_data[0], t_data[0]), 0)
                img = img.reshape([args.batch_size * 2, args.t_length, args.c, args.h, args.w])
                img = img.permute(0, 2, 1, 3, 4)
                img = img.to(device)

                model_output = model(img)

                recons = model_output['output']
                s_re_loss, s_re_loss_per_sample = loss.get_reconstruction_loss(img[:args.batch_size],
                                                                               recons[:args.batch_size], mean=0.5,
                                                                               std=0.5)
                t_re_loss, t_re_loss_per_sample = loss.get_reconstruction_loss(img[args.batch_size:],
                                                                               recons[args.batch_size:], mean=0.5,
                                                                               std=0.5)
                tot_loss = s_re_loss + args.trade_off * t_re_loss
                pretrain_optimizer.zero_grad()
                tot_loss.backward()
                pretrain_optimizer.step()
                if idx == max(len(s_train_batch), len(t_train_batch)) - 1:
                    writer.add_scalar('s_re_loss/AdversarialLossWeight_' + str(args.AdversarialLossWeight), s_re_loss, epoch)
                    writer.add_scalar('t_re_loss/AdversarialLossWeight_' + str(args.AdversarialLossWeight), t_re_loss, epoch)

        torch.save(model.state_dict(), log_dir + '/model-pretrain.pt')
    target_anomaly_label_total = list()
    target_recon_loss_total = list()
    source_recon_loss_total = list()

    with torch.no_grad():
        model.eval()
        for idx in range(len(s_train_batch)):
            s_data = s_train_batch[idx]

            img = s_data[0]
            img = img.reshape([args.batch_size, args.t_length, args.c, args.h, args.w])
            img = img.permute(0, 2, 1, 3, 4)
            img = img.to(device)

            model_output = model(img)

            recons = model_output['output']

            re_loss, re_loss_per_sample = loss.get_reconstruction_loss(img, recons, mean=0.5, std=0.5)

            source_recon_loss_total.append(torch.squeeze(re_loss_per_sample))

        for idx in range(len(t_train_batch)):
            t_data = t_train_batch[idx]

            img = t_data[0]
            img = img.reshape([args.batch_size, args.t_length, args.c, args.h, args.w])
            img = img.permute(0, 2, 1, 3, 4)
            img = img.to(device)

            model_output = model(img)

            recons = model_output['output']
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
        writer.add_figure('pretrain_recon_kde/AdversarialLossWeight_' + str(args.AdversarialLossWeight), fig)

    optimizer = torch.optim.Adam([
        {'params': model.encoder.parameters()},
        {'params': model.decoder.parameters()},
        {'params': model.mem_rep.parameters()},
        {'params': model.adaptiveAvgPool.parameters()},
        {'params': model.adnet0.parameters()}
    ], lr=args.lr)
    for epoch in range(args.epochs):
        print('epoch/total epoch:' + str(epoch) + '/' + str(args.epochs))
        model.train()
        optimizer = exp_lr_scheduler(optimizer, epoch+args.epochs, init_lr=args.lr,
                                     lr_decay_step=args.lr_decay_step,
                                     step_decay_weight=args.step_decay_weight)
        tr_re_loss, tr_ad_loss0, tr_tot = 0.0, 0.0, 0.0

        target_anomaly_label_total = list()
        target_recon_loss_total = list()
        target_domain_prediction0_orig_total = list()

        source_recon_loss_total = list()
        source_domain_prediction0_orig_total = list()

        for idx in range(max(len(s_train_batch), len(t_train_batch))):
            iter_num = epoch * max(len(s_train_batch), len(t_train_batch)) + idx
            total_iter_num = args.epochs * max(len(s_train_batch), len(t_train_batch))
            model.adnet0.coeff = np.float(2.0 * args.u / (1.0 + np.exp(-args.alpha * iter_num / total_iter_num)) - args.u)
            beta = np.float(2.0*(args.beta_high-args.beta_low)/(1.0+np.exp(-1.0*iter_num/total_iter_num))-(args.beta_high-args.beta_low)+args.beta_low)
            if idx == max(len(s_train_batch), len(t_train_batch)) - 1:
                writer.add_scalar('model.adnet0.coeff/AdversarialLossWeight_' + str(args.AdversarialLossWeight), model.adnet0.coeff, iter_num)
                writer.add_scalar('beta/AdversarialLossWeight_' + str(args.AdversarialLossWeight), beta, iter_num)

            s_idx = idx % len(s_train_batch)
            t_idx = idx % len(t_train_batch)

            s_data = s_train_batch[s_idx]
            t_data = t_train_batch[t_idx]

            img = torch.cat((s_data[0], t_data[0]), 0)
            img = img.reshape([args.batch_size * 2, args.t_length, args.c, args.h, args.w])
            img = img.permute(0, 2, 1, 3, 4)
            domain_label = torch.tensor([0] * len(s_data[0]) + [1] * len(t_data[0]), dtype=torch.float).view(-1, 1)

            img = img.to(device)
            domain_label = domain_label.to(device)

            model_output = model(img)
            recons, y0, y0_orig = model_output['output'], model_output['y0'], model_output['y0_orig']
            print('y0:', torch.squeeze(y0))
            print('anomaly_label:', torch.squeeze(torch.cat((s_data[1], t_data[1]), 0)))

            # calculate loss

            re_loss, re_loss_per_sample = loss.get_reconstruction_loss(img, recons, mean=0.5, std=0.5,
                                                                       is_weighted_recon=True,
                                                                       lambda_value=args.lambda_value, beta=beta)

            target_anomaly_label_total.append(torch.squeeze(t_data[1]))
            target_recon_loss_total.append(torch.squeeze(re_loss_per_sample)[args.batch_size:])
            target_domain_prediction0_orig_total.append(torch.squeeze(y0_orig[args.batch_size:]))

            source_recon_loss_total.append(torch.squeeze(re_loss_per_sample)[:args.batch_size])
            source_domain_prediction0_orig_total.append(torch.squeeze(y0_orig[:args.batch_size]))

            # calculate weight

            weight = torch.sigmoid(args.lambda_value * re_loss_per_sample[args.batch_size:] + beta)
            weight = weight.detach().to(device)
            adversarial_weight = torch.cat([torch.tensor([1.0] * args.batch_size).to(device), weight], 0)
            adversarial_weight = torch.unsqueeze(adversarial_weight, 1)
            adversarial_weight = adversarial_weight.detach().to(device)

            print('adversarial weight:', adversarial_weight)
            print('BCELoss:', nn.BCELoss(reduction='none')(y0, domain_label))
            print('----------------------------------------------------------------------------------')

            ad_loss0 = torch.mean(adversarial_weight * nn.BCELoss(reduction='none')(y0, domain_label))
            tot_loss = re_loss + args.AdversarialLossWeight * ad_loss0

            tr_re_loss += re_loss.data.item()
            tr_ad_loss0 += ad_loss0.data.item()
            tr_tot += tot_loss.data.item()

            optimizer.zero_grad()
            tot_loss.backward()
            optimizer.step()

        if epoch == 0 or epoch % 10 == 9:
            target_anomaly_label_total = torch.cat(target_anomaly_label_total, 0)
            target_recon_loss_total = torch.cat(target_recon_loss_total, 0)
            target_domain_prediction0_orig_total = torch.cat(target_domain_prediction0_orig_total, 0)
            source_recon_loss_total = torch.cat(source_recon_loss_total, 0)
            source_domain_prediction0_orig_total = torch.cat(source_domain_prediction0_orig_total, 0)

            target_anomaly_label_total = target_anomaly_label_total.detach().cpu()
            target_recon_loss_total = target_recon_loss_total.detach().cpu()
            target_domain_prediction0_orig_total = target_domain_prediction0_orig_total.detach().cpu()
            source_recon_loss_total = source_recon_loss_total.detach().cpu()
            source_domain_prediction0_orig_total = source_domain_prediction0_orig_total.detach().cpu()

            target_normal_mask = target_anomaly_label_total.eq(0)
            target_abnormal_mask = target_anomaly_label_total.eq(1)

            target_normal_recon = torch.masked_select(target_recon_loss_total, target_normal_mask)
            target_abnormal_recon = torch.masked_select(target_recon_loss_total, target_abnormal_mask)

            target_normal_prediction0 = torch.masked_select(target_domain_prediction0_orig_total, target_normal_mask)
            target_abnormal_prediction0 = torch.masked_select(target_domain_prediction0_orig_total, target_abnormal_mask)

            # recon distribution
            max_length = max(len(target_normal_recon), len(target_abnormal_recon), len(source_recon_loss_total))
            if len(target_normal_recon) == max_length:
                source_recon_loss_total = torch.cat(
                    [source_recon_loss_total.cpu(),
                     torch.tensor([np.nan] * (max_length - len(source_recon_loss_total)))])
                target_abnormal_recon = torch.cat(
                    [target_abnormal_recon.cpu(), torch.tensor([np.nan] * (max_length - len(target_abnormal_recon)))])
            elif len(target_abnormal_recon) == max_length:
                source_recon_loss_total = torch.cat(
                    [source_recon_loss_total.cpu(),
                     torch.tensor([np.nan] * (max_length - len(source_recon_loss_total)))])
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
                    'source_recon': source_recon_loss_total
                }
            )
            fig, ax = plt.subplots()
            sns.kdeplot(data=df, fill=True, common_norm=False, palette='crest', alpha=.5, linewidth=0)
            fig.savefig(os.path.join(log_dir, 'recon_kde.png'))
            writer.add_figure('recon_kde/AdversarialLossWeight_' + str(args.AdversarialLossWeight) + '/epoch_' + str(epoch), fig)

            # domain discriminator0 distribution
            max_length = max(len(target_normal_prediction0), len(target_abnormal_prediction0), len(source_domain_prediction0_orig_total))
            if len(target_normal_prediction0) == max_length:
                source_domain_prediction0_orig_total = torch.cat(
                    [source_domain_prediction0_orig_total.cpu(),
                     torch.tensor([np.nan] * (max_length - len(source_domain_prediction0_orig_total)))])
                target_abnormal_prediction0 = torch.cat(
                    [target_abnormal_prediction0.cpu(), torch.tensor([np.nan] * (max_length - len(target_abnormal_prediction0)))])
            elif len(target_abnormal_prediction0) == max_length:
                source_domain_prediction0_orig_total = torch.cat(
                    [source_domain_prediction0_orig_total.cpu(),
                     torch.tensor([np.nan] * (max_length - len(source_domain_prediction0_orig_total)))])
                target_normal_prediction0 = torch.cat(
                    [target_normal_prediction0.cpu(), torch.tensor([np.nan] * (max_length - len(target_normal_prediction0)))])
            elif len(source_domain_prediction0_orig_total) == max_length:
                target_normal_prediction0 = torch.cat(
                    [target_normal_prediction0.cpu(), torch.tensor([np.nan] * (max_length - len(target_normal_prediction0)))])
                target_abnormal_prediction0 = torch.cat(
                    [target_abnormal_prediction0.cpu(), torch.tensor([np.nan] * (max_length - len(target_abnormal_prediction0)))])
            else:
                raise Exception('wrong length')
            df = pd.DataFrame(
                {
                    'target_normal_prediction0': target_normal_prediction0,
                    'target_abnormal_prediction0': target_abnormal_prediction0,
                    'source_prediction0': source_domain_prediction0_orig_total
                }
            )
            fig, ax = plt.subplots()
            sns.kdeplot(data=df, fill=True, common_norm=False, palette='crest', alpha=.5, linewidth=0)
            fig.savefig(os.path.join(log_dir, 'prediction0_kde.png'))
            writer.add_figure('prediction0_kde/AdversarialLossWeight_' + str(args.AdversarialLossWeight) + '/epoch_' + str(epoch), fig)

        writer.add_scalar('tr_re_loss/AdversarialLossWeight_' + str(args.AdversarialLossWeight),
                          tr_re_loss / max(len(s_train_batch), len(t_train_batch)),
                          epoch)
        writer.add_scalar('tr_ad_loss0/AdversarialLossWeight_' + str(args.AdversarialLossWeight),
                          tr_ad_loss0 / max(len(s_train_batch), len(t_train_batch)),
                          epoch)
        writer.add_scalar('tr_tot/AdversarialLossWeight_' + str(args.AdversarialLossWeight),
                          tr_tot / max(len(s_train_batch), len(t_train_batch)), epoch)

else:
    raise Exception('invaild Model Name')

sys.stdout = orig_stdout
writer.close()
f.close()

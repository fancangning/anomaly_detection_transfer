#!/usr/bin/python
#****************************************************************#
# ScriptName: deAE_test.py
# Author: fancangning.fcn@alibaba-inc.com
# Create Date: 2021-03-18 17:30
# Modify Author: fancangning.fcn@alibaba-inc.com
# Modify Date: 2021-03-18 17:30
# Function: deAE test
#***************************************************************#

import torch
from torch import nn
import os
import sys
import numpy as np
import time
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader
import torch.utils.data as data
from models import AutoEncoderCov2D, AutoEncoderCov2DMem, AutoEncoderCov2DMemFace, AdversarialAutoEncoderCov2D, DoubleEncoderAutoEncoderCov2D
import data.utils as data_utils
import models.loss as loss
from sklearn import metrics
import argparse
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
import pandas as pd
import seaborn as sns


def calculationAUC_plot(fpr, tpr, log_dir, args, recon_type):
    auc = metrics.auc(fpr, tpr)

    # plot
    f, ax = plt.subplots()
    plt.plot(fpr, tpr, label='ROC(area = {0:.4f})'.format(auc), lw=2)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(log_dir, args.source_dataset + 'to' + args.target_dataset + '_' + recon_type + 'roc_curve.png'))

parser = argparse.ArgumentParser(description='Memorizing_Normality')
parser.add_argument('--gpu', type=str, default='0', help='the gpu to use')
parser.add_argument('--source_dataset', type=str, default='USPS', help='source dataset')
parser.add_argument('--target_dataset', type=str, default='MNIST', help='target dataset')
parser.add_argument('--dataset_path', type=str, default='../data/')
parser.add_argument('--version', type=int, default=0, help='experiment version')
parser.add_argument('--ckpt_step', type=int, default=79)
parser.add_argument('--AdversarialLossWeight', type=float, default=0.1)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--exp_dir', type=str, default='./deAE_log/')
parser.add_argument('--h', type=int, default=28, help='height of input images')
parser.add_argument('--w', type=int, default=28, help='width of input images')
parser.add_argument('--c', type=int, default=1, help='channel of input images')
parser.add_argument('--batch_size', type=int, default=1, help='batch size for testing')
parser.add_argument('--num_workers', type=int, default=1, help='number of workers for the test loader')
parser.add_argument('--ModelName', help='AE/MemAE/MemAEFace/AdversarialAE/DoubleEncoderAE', type=str, default='DoubleEncoderAE')
parser.add_argument('--fine_tune', help='whether fine tune', type=bool, default=False)
parser.add_argument('--anomaly_rate', help='the rate of anomaly', type=float, default=0.25)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if args.fine_tune:
    log_dir = os.path.join(args.exp_dir, args.source_dataset+'to'+args.target_dataset+'_'+args.ModelName+'_finetune',
                           'lr_%.5f_adversariallossweight_%.5f_anomalyrate_%.5f_version_%d'%(args.lr, args.AdversarialLossWeight, args.anomaly_rate, args.version))
else:
    log_dir = os.path.join(args.exp_dir, args.source_dataset+'to'+args.target_dataset+'_'+args.ModelName,
                           'lr_%.5f_adversariallossweight_%.5f_anomalyrate_%.5f_version_%d'%(args.lr, args.AdversarialLossWeight, args.anomaly_rate, args.version))
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

orig_stdout = sys.stdout
f = open(os.path.join(log_dir, 'output_%s_%d.txt' % ('original_1.00', args.ckpt_step)), 'w')
sys.stdout = f

model_dir = os.path.join(log_dir, 'model-00%d.pt' % args.ckpt_step)

# Prepare dataset

if args.source_dataset == 'celeb10w' and args.target_dataset == 'celeb10w':
    img_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, )) # [0, 1] -> [-1, 1]
    ])
elif args.source_dataset == 'MNIST' and args.target_dataset == 'MNIST':
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, )) # [0, 1] -> [-1, 1]
    ])
elif args.source_dataset == 'MNIST' and args.target_dataset == 'SVHN':
    img_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, )) # [0, 1] -> [-1, 1]
    ])
elif args.source_dataset == 'SVHN' and args.target_dataset == 'SVHN':
    img_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, )) # [0, 1] -> [-1, 1]
    ])
elif args.source_dataset == 'crop_SVHN' and args.target_dataset == 'crop_SVHN':
    img_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, )) # [0, 1] -> [-1, 1]
    ])
elif args.source_dataset == 'USPS' and args.target_dataset == 'USPS':
    img_transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, )) # [0, 1] -> [-1, 1]
    ])
elif args.source_dataset == 'MNIST' and args.target_dataset == 'USPS':
    img_transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, )) # [0, 1] -> [-1, 1]
    ])
elif args.source_dataset == 'USPS' and args.target_dataset == 'MNIST':
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, )) # [0, 1] -> [-1, 1]
    ])
else:
    raise Exception('No such source dataset')

if args.target_dataset == 'MNIST':
    t_dataset = torchvision.datasets.MNIST(args.dataset_path, download=True, train=False, transform=img_transform)
elif args.target_dataset == 'SVHN':
    t_dataset = torchvision.datasets.SVHN(args.dataset_path, download=True, split='test', transform=img_transform)
elif args.target_dataset == 'celeb10w':
    dataset = data_utils.Celeb10wDataset(args.dataset_path, transform=img_transform)
    train, val, test = data.random_split(dataset, [50000, 10000, 11114], generator=torch.Generator().manual_seed(2020))
elif args.target_dataset == 'crop_SVHN':
    t_dataset = torchvision.datasets.ImageFolder(os.path.join(args.dataset_path, 'test_crop'), transform=img_transform)
elif args.target_dataset == 'USPS':
    t_dataset = torchvision.datasets.USPS(args.dataset_path, train=False, transform=img_transform, download=True)
else:
    raise Exception('invalid target dataset')
t_dataset = data_utils.AnomalyDataset(t_dataset, [0], args.anomaly_rate)

t_batch = data.DataLoader(t_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)

# Model setting
if args.ModelName == 'DoubleEncoderAE':
    model = DoubleEncoderAutoEncoderCov2D(args.c, 0.0)
else:
    raise Exception('Wrong model name')

model_para = torch.load(model_dir)
model.load_state_dict(model_para)
model.to(device)
model.eval()

# Test

img_recon_error_list = list()
embedding_recon_error_list = list()
anomaly_label = list()

for batch_idz, data in enumerate(t_batch):
    label = data[1]
    img = data[0].to(device)
    model_output = model(img)
    recons, y0, y0_orig, f, f2 = model_output['recons'], model_output['y0'], model_output['y0_orig'], model_output['f'], model_output['f2']
    img_re_loss, img_re_loss_per_sample = loss.get_reconstruction_loss(img, recons, mean=0.5, std=0.5)
    embedding_re_loss, embedding_re_loss_per_sample = loss.get_reconstruction_loss(f, f2, mean=0.5, std=0.5)

    anomaly_label += label.tolist()
    img_recon_error_list.append(float(img_re_loss.cpu()))
    embedding_recon_error_list.append(float(embedding_re_loss.cpu()))

print('anomaly_label:', np.array(anomaly_label))
print('img_recon_error_list:', np.array(img_recon_error_list))
print('embedding_recon_error_list:', np.array(embedding_recon_error_list))

fpr, tpr, thresholds = metrics.roc_curve(np.array(anomaly_label), np.array(img_recon_error_list))
calculationAUC_plot(fpr, tpr, log_dir, args, 'img_recon')
fpr, tpr, thresholds = metrics.roc_curve(np.array(anomaly_label), np.array(embedding_recon_error_list))
calculationAUC_plot(fpr, tpr, log_dir, args, 'embedding_recon')

sys.stdout = orig_stdout

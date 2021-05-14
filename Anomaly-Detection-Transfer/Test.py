#!/usr/bin/python
#****************************************************************#
# ScriptName: Test.py
# Author: fancangning.fcn@alibaba-inc.com
# Create Date: 2021-01-26 20:33
# Modify Author: fancangning.fcn@alibaba-inc.com
# Modify Date: 2021-01-26 20:33
# Function: Test
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
from models import AutoEncoderCov2D, AutoEncoderCov2DMem, AutoEncoderCov2DMemFace, AdversarialAutoEncoderCov2D
import data.utils as data_utils
import models.loss as loss
from sklearn import metrics
import argparse
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
import pandas as pd
import seaborn as sns

parser = argparse.ArgumentParser(description='Memorizing_Normality')
parser.add_argument('--gpu', type=str, default='0', help='the gpu to use')
parser.add_argument('--source_dataset', type=str, default='MNIST', help='source dataset')
parser.add_argument('--target_dataset', type=str, default='USPS', help='target dataset')
parser.add_argument('--dataset_path', type=str, default='../data/')
parser.add_argument('--version', type=int, default=0, help='experiment version')
parser.add_argument('--ckpt_step', type=str, default='0099')
parser.add_argument('--AdversarialLossWeight', type=float, default=0.1)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--exp_dir', type=str, default='./log/')
parser.add_argument('--h', type=int, default=28, help='height of input images')
parser.add_argument('--w', type=int, default=28, help='width of input images')
parser.add_argument('--c', type=int, default=3, help='channel of input images')
parser.add_argument('--batch_size', type=int, default=1, help='batch size for testing')
parser.add_argument('--num_workers', type=int, default=1, help='number of workers for the test loader')
parser.add_argument('--ModelName', help='AE/MemAE/MemAEFace/AdversarialAE', type=str, default='AdversarialAE')
parser.add_argument('--MemDim', help='Memory Dimention', type=int, default=100)
parser.add_argument('--ShrinkThres', help='ShrinkThres', type=float, default=0.025)
parser.add_argument('--kde', help='whether draw the kde', type=bool, default=True)
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
f = open(os.path.join(log_dir, 'output_%s_%s.txt' % ('original_1.00', args.ckpt_step)), 'w')
sys.stdout = f

model_dir = os.path.join(log_dir, 'model-%s.pt' % args.ckpt_step)

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
        data_utils.ToThreeChannel(),
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

if args.ModelName == 'AE':
    model = AutoEncoderCov2D(args.c)
elif args.ModelName == 'MemAE':
    model = AutoEncoderCov2DMem(args.c, args.MemDim, shrink_thres=args.ShrinkThres)
elif args.ModelName == 'MemAEFace':
    model = AutoEncoderCov2DMemFace(args.c, args.MemDim, shrink_thres=args.ShrinkThres)
elif args.ModelName == 'AdversarialAE':
    model = AdversarialAutoEncoderCov2D(args.c, backward_coeff=0.0)
else:
    raise Exception('Wrong model name')

model_para = torch.load(model_dir)
model.load_state_dict(model_para)
model.to(device)
model.eval()

# Test

recon_error_list = list()
anomaly_label = list()
y_total = list()
if args.ModelName == 'AdversarialAE':
    for batch_idx, data in enumerate(t_batch):
        label = data[1]
        img = data[0].to(device)
        model_output = model(img)
        recons, domain_prediction, domain_prediction0 = model_output['out'], model_output['y'], model_output['y0']
        re_loss, re_loss_per_sample = loss.get_reconstruction_loss(img, recons, mean=0.5, std=0.5)
        print('label:', label)
        print('y:', domain_prediction[0])
        print('re_loss:', re_loss)
        anomaly_label += label.tolist()
        y_total += domain_prediction[0].cpu().tolist()
        recon_error_list.append(float(re_loss.cpu()))
else:
    for batch_idx, data in enumerate(t_batch):
        label = data[1]
        img = data[0].to(device)
        model_output = model(img)
        recons, attr = model_output['output'], model_output['att']
        re_loss = loss.get_reconstruction_loss(img, recons, mean=0.5, std=0.5)
        print('label:', label)
        print('re_loss', re_loss)
        anomaly_label += label.tolist()
        recon_error_list.append(float(re_loss.cpu()))

print('anomaly_label:', np.array(anomaly_label))
print('y:', np.array(domain_prediction.detach().cpu()))
print('recon_error_list:', np.array(recon_error_list))

if args.kde:
    normal_recon_error = list()
    abnormal_recon_error = list()
    for i in range(len(anomaly_label)):
        if anomaly_label[i] == 0:
            normal_recon_error.append(recon_error_list[i])
        elif anomaly_label[i] == 1:
            abnormal_recon_error.append(recon_error_list[i])
        else:
            pass
    max_length = max(len(normal_recon_error), len(abnormal_recon_error))
    if len(normal_recon_error) < len(abnormal_recon_error):
        normal_recon_error += [np.nan]*(max_length - len(normal_recon_error))
    elif len(abnormal_recon_error) < len(normal_recon_error):
        abnormal_recon_error += [np.nan]*(max_length - len(abnormal_recon_error))
    else:
        pass
    df = pd.DataFrame(
                {
                    'normal_recon_error': normal_recon_error,
                    'abnormal_recon_error': abnormal_recon_error,
                }
            )
    f, ax = plt.subplots()
    sns.kdeplot(data=df, fill=True, common_norm=False, palette='crest', alpha=.5, linewidth=0)
    f.savefig(os.path.join(log_dir, 'kde.png'))

fpr, tpr, thresholds = metrics.roc_curve(np.array(anomaly_label), np.array(recon_error_list))
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
plt.savefig(os.path.join(log_dir, args.source_dataset+'to'+args.target_dataset+'_roc_curve.png'))


sys.stdout = orig_stdout
